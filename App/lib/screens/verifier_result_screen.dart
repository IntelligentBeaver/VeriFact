import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/models/verifier_model.dart';
import 'package:verifact_app/utils/constants/colors.dart';
import 'package:verifact_app/utils/constants/sizes.dart';
import 'package:verifact_app/utils/notifiers/verifier_notifier.dart';
import 'package:verifact_app/widgets/results/custom_source_card.dart';

class VerifierResultScreen extends ConsumerStatefulWidget {
  const VerifierResultScreen({required this.claim, super.key});

  final String claim;

  @override
  ConsumerState<VerifierResultScreen> createState() =>
      _VerifierResultScreenState();
}

class _VerifierResultScreenState extends ConsumerState<VerifierResultScreen> {
  @override
  void initState() {
    super.initState();
    // Trigger verification on open
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(verifierProvider.notifier).verify(widget.claim);
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(verifierProvider);

    return Scaffold(
      backgroundColor: context.color.surfaceBright,
      appBar: AppBar(
        title: const Text('Analysis Result'),
        centerTitle: false,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: state.when(
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (e, _) => Center(child: Text(e.toString())),
          data: (VerifierModel? data) {
            if (data == null) return const Center(child: Text('No result'));

            // Use first evidence item (if any) for image/text preview
            final Evidence? _firstEvidence =
                (data.evidence != null && data.evidence!.isNotEmpty)
                ? data.evidence!.first
                : null;

            // Helper to detect an image URL (simple heuristics)
            bool hasImage() {
              final url = _firstEvidence?.url ?? '';
              return url.isNotEmpty &&
                  (url.endsWith('.png') ||
                      url.endsWith('.jpg') ||
                      url.endsWith('.jpeg') ||
                      url.endsWith('.webp'));
            }

            return SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Top card: confidence circle + verdict + explain
                  Card(
                    color: context.color.surface,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(
                        AppSizes.borderRadiusLg,
                      ),
                    ),
                    elevation: 2,
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        children: [
                          // Confidence ring
                          _ConfidenceRing(
                            percent: (data.confidence ?? 0.0) * 100.0,
                          ),
                          SizedBox(height: AppSizes.mdLg),
                          // Verdict pill + Explain why
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              _VerdictPill(
                                verdict: data.verdict,
                                onTap: () {
                                  showDialog<void>(
                                    context: context,
                                    builder: (ctx) => AlertDialog(
                                      title: const Text('Label scores'),
                                      content: Column(
                                        mainAxisSize: MainAxisSize.min,
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            'Supports: ${(data.scores.supports).toStringAsFixed(2)}',
                                          ),
                                          SizedBox(height: 8),
                                          Text(
                                            'Refutes: ${(data.scores.refutes).toStringAsFixed(2)}',
                                          ),
                                          SizedBox(height: 8),
                                          Text(
                                            'Neutral: ${(data.scores.neutral).toStringAsFixed(2)}',
                                          ),
                                        ],
                                      ),
                                      actions: [
                                        TextButton(
                                          onPressed: () => Navigator.pop(ctx),
                                          child: const Text('Close'),
                                        ),
                                      ],
                                    ),
                                  );
                                },
                              ),
                            ],
                          ),
                          SizedBox(height: AppSizes.smMd),
                          _ExplainPill(
                            onTap: () {
                              // simple placeholder behaviour
                              // showModalBottomSheet<void>(
                              //   context: context,
                              //   builder: (ctx) => Padding(
                              //     padding: const EdgeInsets.all(16.0),
                              //     child: Text(
                              //       _firstEvidence?.text ??
                              //           'No explanation available',
                              //     ),
                              //   ),
                              // );
                            },
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Image + extracted text row
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (hasImage()) ...[
                        Expanded(
                          child: _SourceImageCard(
                            imageUrl: _firstEvidence!.url,
                          ),
                        ),
                        const SizedBox(width: 12),
                      ],
                      Expanded(
                        flex: 2,
                        child: _ExtractedTextCard(
                          text: _firstEvidence?.text ?? 'No extracted text',
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 20),

                  // Scientific Evidence section
                  Row(
                    spacing: AppSizes.xsSm,
                    children: [
                      Icon(
                        LucideIcons.bookOpen500,
                        size: 18.r,
                        color: context.color.primary,
                      ),
                      Expanded(
                        child: Text(
                          'Scientific Evidence',
                          style: context.text.headlineLarge?.copyWith(
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (data.evidence != null && data.evidence!.isNotEmpty)
                    ...data.evidence!.map(
                      (e) => Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: CustomSourceCard(
                          title: e.title.isEmpty ? 'Source' : e.title,
                          text: e.text,
                          url: e.url,
                          score: e.score,
                        ),
                      ),
                    ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

class _ConfidenceRing extends StatelessWidget {
  const _ConfidenceRing({required this.percent});
  final double percent;

  @override
  Widget build(BuildContext context) {
    final display = percent.clamp(0.0, 100.0).round();
    return SizedBox(
      height: 140,
      width: 140,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Container(
            height: 140,
            width: 140,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(
                color: context.color.surfaceVariant,
                width: 6,
              ),
            ),
          ),
          // progress arc — simplified by using CircularProgressIndicator
          SizedBox(
            height: 140,
            width: 140,
            child: CircularProgressIndicator(
              value: (percent / 100).clamp(0.0, 1.0),
              strokeWidth: 12,
              color: context.color.error,
              backgroundColor: context.color.surfaceVariant,
            ),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '$display%',
                style: context.text.displayMedium?.copyWith(
                  fontWeight: FontWeight.w800,
                ),
              ),
              SizedBox(height: 4),
              Text('CONFIDENCE', style: context.text.labelSmall),
            ],
          ),
        ],
      ),
    );
  }
}

class _VerdictPill extends StatelessWidget {
  const _VerdictPill({required this.verdict, required this.onTap});
  final String verdict;
  final VoidCallback onTap;

  Color _bgColor(BuildContext context) {
    switch (verdict.toLowerCase()) {
      case 'supports':
        return AppColors.success.withAlpha(42);
      case 'refutes':
        return context.color.errorContainer;
      default:
        return context.color.surfaceContainer;
    }
  }

  Color _textColor(BuildContext context) {
    switch (verdict.toLowerCase()) {
      case 'supports':
        return AppColors.success;
      case 'refutes':
        return context.color.error;
      default:
        return context.color.onSurfaceVariant;
    }
  }

  IconData _verdictIcon() {
    switch (verdict.toLowerCase()) {
      case 'supports':
        return LucideIcons.circleCheck;
      case 'refutes':
        return LucideIcons.circleX;
      default:
        return LucideIcons.circleMinus;
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: AppSizes.lg,
          vertical: AppSizes.smMd,
        ),
        decoration: BoxDecoration(
          color: _bgColor(context),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              _verdictIcon(),
              size: AppSizes.iconSm,
              color: _textColor(context),
            ),
            const SizedBox(width: 8),
            Text(
              verdict.toUpperCase(),
              style: context.text.bodyLarge?.copyWith(
                fontWeight: FontWeight.w900,
                color: _textColor(context),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ExplainPill extends StatelessWidget {
  const _ExplainPill({required this.onTap});
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: context.color.primaryContainer,
          borderRadius: BorderRadius.circular(999),
        ),
        child: Text(
          'View Evidence',
          style: context.text.labelLarge?.copyWith(
            color: context.color.primary,
          ),
        ),
      ),
    );
  }
}

class _SourceImageCard extends StatelessWidget {
  const _SourceImageCard({required this.imageUrl});
  final String imageUrl;

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: SizedBox(
        height: 140,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.network(
            imageUrl,
            fit: BoxFit.cover,
            errorBuilder: (_, __, ___) => const SizedBox.shrink(),
          ),
        ),
      ),
    );
  }
}

class _ExtractedTextCard extends StatelessWidget {
  const _ExtractedTextCard({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: context.color.surface,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Extracted Text',
              style: context.text.labelSmall?.copyWith(
                fontWeight: FontWeight.w700,
                color: context.color.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              text,
              style: context.text.labelSmall?.copyWith(
                fontWeight: FontWeight.w800,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

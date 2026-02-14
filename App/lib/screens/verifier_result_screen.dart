import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/models/history_record.dart';
import 'package:verifact_app/models/verifier_model.dart';
import 'package:verifact_app/services/history_service.dart';
import 'package:verifact_app/utils/constants/colors.dart';
import 'package:verifact_app/utils/constants/sizes.dart';
import 'package:verifact_app/utils/helpers/device_utility.dart';
import 'package:verifact_app/utils/notifiers/verifier_notifier.dart';
import 'package:verifact_app/widgets/results/custom_source_card.dart';

class VerifierResultScreen extends ConsumerStatefulWidget {
  const VerifierResultScreen({
    required this.claim,
    this.initialData,
    super.key,
  });

  final String claim;
  final VerifierModel? initialData;

  @override
  ConsumerState<VerifierResultScreen> createState() =>
      _VerifierResultScreenState();
}

class _VerifierResultScreenState extends ConsumerState<VerifierResultScreen> {
  bool _saved = false;
  VerifierModel? _initialData;
  @override
  void initState() {
    super.initState();
    // Use provided initialData when available, otherwise trigger verification
    if (widget.initialData != null) {
      _initialData = widget.initialData;
      _saved = true;
    } else {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        ref.read(verifierProvider.notifier).verify(widget.claim);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(verifierProvider);

    return Scaffold(
      backgroundColor: DeviceUtility.isDarkMode(context)
          ? context.theme.scaffoldBackgroundColor
          : context.color.surfaceBright,
      appBar: AppBar(
        title: const Text('Analysis Result'),
        centerTitle: false,
      ),
      body: Padding(
        padding: EdgeInsets.all(AppSizes.smMd),
        child: state.when(
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (e, _) => Center(child: Text(e.toString())),
          data: (VerifierModel? data) {
            final model = data ?? _initialData;
            if (model == null) return const Center(child: Text('No result'));

            // Persist verifier result once
            if (!_saved && data != null) {
              try {
                final firstEvidence =
                    (data.evidence != null && data.evidence!.isNotEmpty)
                    ? data.evidence!.first
                    : null;
                final rec = HistoryRecord(
                  type: 'verifier',
                  query: widget.claim,
                  resultStatus: data.verdict,
                  conclusion: firstEvidence?.text ?? data.verdict,
                  payload: data.toString(),
                  timestamp: DateTime.now().millisecondsSinceEpoch,
                );
                HistoryService().addRecord(rec);
              } catch (_) {}
              _saved = true;
            }

            // Use first evidence item (if any) for image/text preview
            final firstEvidence =
                (model.evidence != null && model.evidence!.isNotEmpty)
                ? model.evidence!.first
                : null;

            // Helper to detect an image URL (simple heuristics)
            bool hasImage() {
              final url = firstEvidence?.url ?? '';
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
                      padding: EdgeInsets.all(AppSizes.md),
                      child: Stack(
                        clipBehavior: Clip.none,
                        children: [
                          Column(
                            children: [
                              // Confidence ring
                              _ConfidenceRing(
                                percent: (model.confidence ?? 0.0) * 100.0,
                              ),
                              SizedBox(height: AppSizes.mdLg),
                              // Verdict pill + Explain why
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  _VerdictPill(
                                    verdict: model.verdict,
                                    onTap: () async {
                                      await showDialog<void>(
                                        context: context,
                                        builder: (ctx) => AlertDialog(
                                          title: const Text('Label scores'),
                                          content: Column(
                                            mainAxisSize: MainAxisSize.min,
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                'Supports: ${model.scores.supports.toStringAsFixed(2)}',
                                              ),
                                              const SizedBox(height: 8),
                                              Text(
                                                'Refutes: ${model.scores.refutes.toStringAsFixed(2)}',
                                              ),
                                              const SizedBox(height: 8),
                                              Text(
                                                'Neutral: ${model.scores.neutral.toStringAsFixed(2)}',
                                              ),
                                            ],
                                          ),
                                          actions: [
                                            TextButton(
                                              onPressed: () =>
                                                  Navigator.pop(ctx),
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
                                onTap: () async {
                                  // simple placeholder behaviour
                                  await showModalBottomSheet<void>(
                                    context: context,
                                    builder: (ctx) => Padding(
                                      padding: EdgeInsets.all(AppSizes.md),
                                      child: Text(
                                        firstEvidence?.text ??
                                            'No explanation available',
                                      ),
                                    ),
                                  );
                                },
                              ),
                            ],
                          ),

                          // Info icon top-right
                          Positioned(
                            top: -10,
                            right: -10,
                            child: IconButton(
                              icon: Icon(
                                Icons.info_outline,
                                size: AppSizes.iconMd,
                                color: context.color.surfaceTint,
                              ),
                              onPressed: () async {
                                await showDialog<void>(
                                  context: context,
                                  builder: (ctx) => AlertDialog(
                                    title: const Text('Model Warnings'),
                                    content: const Column(
                                      mainAxisSize: MainAxisSize.min,
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          'The verifier model provides automated predictions based on available evidence. It can be incorrect, biased, or incomplete.',
                                        ),
                                        SizedBox(height: 8),
                                        Text(
                                          'Do not treat these results as medical, legal, or professional advice. Always consult a qualified professional for health-related claims.',
                                        ),
                                        SizedBox(height: 8),
                                        Text(
                                          'Use the provided sources and original documents to verify claims; when in doubt seek expert guidance.',
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
                            imageUrl: firstEvidence!.url,
                          ),
                        ),
                        const SizedBox(width: 12),
                      ],
                      Expanded(
                        flex: 2,
                        child: _ExtractedTextCard(
                          text: widget.claim.isNotEmpty
                              ? widget.claim
                              : (firstEvidence?.text ?? 'No extracted text'),
                        ),
                      ),
                    ],
                  ),

                  SizedBox(height: AppSizes.md),

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
                  SizedBox(height: AppSizes.md),

                  if (model.evidence != null && model.evidence!.isNotEmpty)
                    ...model.evidence!.map(
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
                color: context.color.surfaceContainerHighest,
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
              backgroundColor: context.color.surfaceContainerHighest,
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
              const SizedBox(height: 4),
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
        padding: const EdgeInsets.all(12),
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

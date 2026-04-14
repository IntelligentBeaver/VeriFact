import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/models/verifier_model.dart';
import 'package:verifact_app/screens/history_preview_screen.dart';
import 'package:verifact_app/screens/verifier_result_screen.dart';
import 'package:verifact_app/services/history_service.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class HistoryScreen extends ConsumerStatefulWidget {
  const HistoryScreen({super.key});

  @override
  ConsumerState<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends ConsumerState<HistoryScreen> {
  final HistoryService _svc = HistoryService();

  Future<Map<String, List<RecordWithKey>>>? _groupsFuture;
  String _selectedType = 'all'; // 'all' | 'verifier' | 'qa' | 'doc'

  @override
  void initState() {
    super.initState();
    _groupsFuture = _svc.getGroupedByDay();
  }

  Color _statusColor(BuildContext context, String? status) {
    if (status == null) return context.color.surfaceContainerHighest;
    final s = status.toLowerCase();
    if (s.contains('true') || s.contains('support')) return Colors.green;
    if (s.contains('false') || s.contains('refute')) return Colors.red;
    if (s.contains('mislead') || s.contains('not enough evidence')) {
      return Colors.orange;
    }
    return context.color.surfaceContainerHighest;
  }

  Future<void> _refresh() async {
    setState(() {
      _groupsFuture = _svc.getGroupedByDay();
    });
    await _groupsFuture;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: context.theme.scaffoldBackgroundColor,
      appBar: AppBar(
        title: Text(
          'History',
          style: context.text.displayLarge?.copyWith(
            fontWeight: FontWeight.w900,
          ),
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: FutureBuilder<Map<String, List<RecordWithKey>>>(
        future: _groupsFuture,
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          final groups = snap.data ?? {};
          if (groups.isEmpty) {
            return Center(
              child: Text('No history yet.', style: context.text.bodyMedium),
            );
          }

          return RefreshIndicator(
            onRefresh: _refresh,
            child: ListView(
              padding: EdgeInsets.all(AppSizes.md),
              children: [
                // Filter chips
                Wrap(
                  spacing: AppSizes.sm,
                  children: [
                    ChoiceChip(
                      label: const Text('All'),
                      selected: _selectedType == 'all',
                      onSelected: (_) => setState(() => _selectedType = 'all'),
                    ),
                    ChoiceChip(
                      label: const Text('Verifier'),
                      selected: _selectedType == 'verifier',
                      onSelected: (_) =>
                          setState(() => _selectedType = 'verifier'),
                    ),
                    ChoiceChip(
                      label: const Text('Q&A'),
                      selected: _selectedType == 'qa',
                      onSelected: (_) => setState(() => _selectedType = 'qa'),
                    ),
                    ChoiceChip(
                      label: const Text('Retriever'),
                      selected: _selectedType == 'doc',
                      onSelected: (_) => setState(() => _selectedType = 'doc'),
                    ),
                  ],
                ),
                SizedBox(height: AppSizes.md),
                ...groups.entries.map((entry) {
                  // filter records in this day by selected type
                  final filtered = entry.value
                      .where(
                        (rw) =>
                            _selectedType == 'all' ||
                            rw.record.type == _selectedType,
                      )
                      .toList();
                  if (filtered.isEmpty) return const SizedBox.shrink();
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(entry.key, style: context.text.labelLarge),
                      SizedBox(height: AppSizes.sm),
                      ...filtered.map((rw) {
                        final r = rw.record;
                        final dt = DateTime.fromMillisecondsSinceEpoch(
                          r.timestamp,
                        );
                        return Dismissible(
                          key: ValueKey(rw.key),
                          direction: DismissDirection.endToStart,
                          background: Container(
                            alignment: Alignment.centerRight,
                            padding: EdgeInsets.only(right: AppSizes.md),
                            color: Colors.redAccent,
                            child: const Icon(
                              Icons.delete,
                              color: Colors.white,
                            ),
                          ),
                          onDismissed: (_) async {
                            await _svc.deleteRecord(rw.key);
                            await _refresh();
                          },
                          child: Card(
                            margin: EdgeInsets.only(bottom: AppSizes.smMd),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(
                                AppSizes.borderRadiusLg,
                              ),
                            ),
                            child: ListTile(
                              onTap: () async {
                                // Verifier preview
                                if (r.type == 'verifier' && r.payload != null) {
                                  try {
                                    final decoded = jsonDecode(r.payload!);
                                    final json = decoded is Map
                                        ? Map<String, dynamic>.from(decoded)
                                        : <String, dynamic>{};
                                    final model = VerifierModel.fromJson(json);
                                    await Navigator.of(context).push(
                                      MaterialPageRoute<void>(
                                        builder: (_) => VerifierResultScreen(
                                          claim: r.query,
                                          initialData: model,
                                        ),
                                      ),
                                    );
                                    return;
                                  } catch (_) {}
                                }

                                // QA preview
                                if (r.type == 'qa' && r.payload != null) {
                                  await Navigator.of(context).push(
                                    MaterialPageRoute<void>(
                                      builder: (_) =>
                                          HistoryPreviewScreen(record: r),
                                    ),
                                  );
                                  return;
                                }

                                // Retriever / Doc preview
                                if (r.type == 'doc' && r.payload != null) {
                                  await Navigator.of(context).push(
                                    MaterialPageRoute<void>(
                                      builder: (_) =>
                                          HistoryPreviewScreen(record: r),
                                    ),
                                  );
                                  return;
                                }

                                // fallback: open and re-run verification
                                await Navigator.of(context).push(
                                  MaterialPageRoute<void>(
                                    builder: (_) =>
                                        VerifierResultScreen(claim: r.query),
                                  ),
                                );
                              },
                              leading: Container(
                                width: 48,
                                height: 48,
                                decoration: BoxDecoration(
                                  color: context.color.surfaceContainerHighest,
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: Icon(
                                  r.type == 'doc'
                                      ? Icons.description
                                      : r.type == 'qa'
                                      ? Icons.message
                                      : Icons.verified,
                                  color: context.color.onSurfaceVariant,
                                ),
                              ),
                              title: Text(
                                r.query,
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: context.text.bodyMedium,
                              ),
                              subtitle: Text(
                                '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}',
                                style: context.text.bodySmall,
                              ),
                              trailing: Container(
                                padding: EdgeInsets.symmetric(
                                  horizontal: AppSizes.sm,
                                  vertical: AppSizes.xsSm,
                                ),
                                decoration: BoxDecoration(
                                  color: _statusColor(
                                    context,
                                    r.resultStatus,
                                  ).withOpacity(0.12),
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: Text(
                                  r.resultStatus ?? 'Unknown',
                                  style: context.text.labelSmall?.copyWith(
                                    color: _statusColor(
                                      context,
                                      r.resultStatus,
                                    ),
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        );
                      }),
                      SizedBox(height: AppSizes.lg),
                    ],
                  );
                }),
              ],
            ),
          );
        },
      ),
    );
  }
}

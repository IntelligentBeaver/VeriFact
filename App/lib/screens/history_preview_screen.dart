import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/models/history_record.dart';
import 'package:verifact_app/models/qa_model.dart';
import 'package:verifact_app/models/retriever_model.dart';
import 'package:verifact_app/utils/constants/sizes.dart';
import 'package:verifact_app/widgets/results/custom_doc_search_result_card.dart';
import 'package:verifact_app/widgets/results/custom_qa_result_card.dart';
import 'package:verifact_app/widgets/results/custom_results_header.dart';
import 'package:verifact_app/widgets/results/custom_source_card.dart';

class HistoryPreviewScreen extends StatelessWidget {
  const HistoryPreviewScreen({
    required this.record,
    super.key,
  });

  final HistoryRecord record;

  (String conclusion, String evidence) _parseAnswer(String answer) {
    var conclusion = '';
    var evidence = '';

    final conclusionMatch = RegExp(
      r'Conclusion:\s*(.+?)(?=\n\nEvidence:|Evidence:|$)',
      dotAll: true,
    ).firstMatch(answer);
    final evidenceMatch = RegExp(
      r'Evidence:\s*(.+?)$',
      dotAll: true,
    ).firstMatch(answer);

    if (conclusionMatch != null) {
      conclusion = conclusionMatch.group(1)?.trim() ?? '';
    }
    if (evidenceMatch != null) {
      evidence = evidenceMatch.group(1)?.trim() ?? '';
    }
    if (conclusion.isEmpty && evidence.isEmpty) {
      conclusion = answer;
    }
    return (conclusion, evidence);
  }

  @override
  Widget build(BuildContext context) {
    Widget body;

    if (record.type == 'qa' && record.payload != null) {
      try {
        final decoded = jsonDecode(record.payload!);
        final json = decoded is Map
            ? Map<String, dynamic>.from(decoded)
            : <String, dynamic>{};
        final qa = QAModel.fromJson(json);
        final parsed = _parseAnswer(qa.answer);
        body = ListView(
          padding: EdgeInsets.all(AppSizes.md),
          children: [
            Text(
              'Your Answer',
              style: context.text.displayMedium?.copyWith(
                fontWeight: FontWeight.w900,
                color: context.color.onSurface,
              ),
            ),
            SizedBox(height: AppSizes.mdLg),
            CustomQAResultCard(
              conclusion: parsed.$1,
              evidence: parsed.$2,
            ),
            if (qa.sources.isNotEmpty) ...[
              SizedBox(height: AppSizes.md),
              Row(
                children: [
                  Icon(Icons.book, size: 18, color: context.color.primary),
                  SizedBox(width: AppSizes.xsSm),
                  Text(
                    'Sources',
                    style: context.text.headlineLarge?.copyWith(
                      fontWeight: FontWeight.w900,
                      color: context.color.onSurface,
                    ),
                  ),
                ],
              ),
              SizedBox(height: AppSizes.md),
              ...qa.sources.map(
                (s) => Padding(
                  padding: EdgeInsets.only(bottom: AppSizes.smMd),
                  child: CustomSourceCard(
                    title: s.title.isEmpty ? 'Source' : s.title,
                    text: s.text,
                    url: s.url,
                    score: s.score,
                  ),
                ),
              ),
            ],
          ],
        );
      } catch (_) {
        body = Center(
          child: Text(
            'Unable to preview QA result.',
            style: context.text.bodyMedium,
          ),
        );
      }
    } else if (record.type == 'doc' && record.payload != null) {
      try {
        final decoded = jsonDecode(record.payload!);
        final json = decoded is Map
            ? Map<String, dynamic>.from(decoded)
            : <String, dynamic>{};
        final resp = RetrieverResponse.fromJson(json);
        if (resp.results.isEmpty) {
          body = Center(
            child: Text(
              'No documents in preview.',
              style: context.text.bodyMedium,
            ),
          );
        } else {
          body = ListView(
            padding: EdgeInsets.all(AppSizes.md),
            children: [
              const CustomResultsHeader(title: 'Your Results'),
              SizedBox(height: AppSizes.mdLg),
              ...resp.results.map((result) {
                final passage = result.passage;
                final author = passage?.author ?? 'Unknown Author';
                final url = passage?.url;
                final title =
                    passage?.title ?? passage?.sectionHeading ?? 'Result';
                final snippet = (passage?.text ?? '').replaceAll('\n', ' ');
                final score =
                    result.finalScore ?? result.crossScore ?? result.faissScore;
                return Padding(
                  padding: EdgeInsets.only(bottom: AppSizes.smMd),
                  child: CustomDocSearchResultCard(
                    title: title,
                    content: snippet.isEmpty
                        ? 'No preview available.'
                        : (snippet.length > 320
                              ? '${snippet.substring(0, 320)}...'
                              : snippet),
                    author: author,
                    score: score,
                    url: url,
                  ),
                );
              }),
            ],
          );
        }
      } catch (_) {
        final msg =
            (record.payload != null &&
                record.payload!.startsWith('Instance of'))
            ? 'No preview available for this record. Re-run the search to save a preview.'
            : 'Unable to preview documents.';
        body = Center(
          child: Text(
            msg,
            style: context.text.bodyMedium,
          ),
        );
      }
    } else {
      body = Padding(
        padding: EdgeInsets.all(AppSizes.md),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(record.query, style: context.text.titleLarge),
            SizedBox(height: AppSizes.md),
            Text(
              'No preview available for this record.',
              style: context.text.bodyMedium,
            ),
          ],
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Preview', style: context.text.headlineLarge),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: body,
    );
  }
}

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:verifact_app/models/retriever_model.dart';
import 'package:verifact_app/utils/notifiers/qa_notifier.dart';
import 'package:verifact_app/utils/notifiers/retriever_notifier.dart';
import 'package:verifact_app/utils/notifiers/verifier_notifier.dart';

class ResponseTestScreen extends ConsumerStatefulWidget {
  const ResponseTestScreen({super.key});

  @override
  ConsumerState<ResponseTestScreen> createState() => _ResponseTestScreenState();
}

class _ResponseTestScreenState extends ConsumerState<ResponseTestScreen>
    with TickerProviderStateMixin {
  final _qaController = TextEditingController();
  final _qaTopKController = TextEditingController(text: '10');
  final _qaMinScoreController = TextEditingController(text: '0.4');

  final _verifierClaimController = TextEditingController();
  final _verifierBodyController = TextEditingController();
  bool _verifierUseBody = false;

  final _retrieverQueryController = TextEditingController();
  final _retrieverBodyController = TextEditingController();
  bool _retrieverUseBody = false;

  // track current provider keys (null = no request yet)
  String? _currentQaQuestion;

  String? _currentVerifierKey;
  String? _currentRetrieverKey;

  @override
  void dispose() {
    _qaController.dispose();
    _qaTopKController.dispose();
    _qaMinScoreController.dispose();
    _verifierClaimController.dispose();
    _verifierBodyController.dispose();
    _retrieverQueryController.dispose();
    _retrieverBodyController.dispose();
    super.dispose();
  }

  Widget _skeletonizer() {
    return Column(
      children: List.generate(
        3,
        (i) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0),
          child: Container(
            height: 80,
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Padding(
              padding: EdgeInsets.all(12.0),
              child: Align(
                alignment: Alignment.centerLeft,
                child: LinearProgressIndicator(),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _resultCard(AsyncValue<dynamic> state) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 400),
      switchInCurve: Curves.easeOut,
      switchOutCurve: Curves.easeIn,
      child: state.when(
        data: (data) {
          if (data == null) {
            return Card(
              key: const ValueKey('data'),
              elevation: 2,
              child: const Padding(
                padding: EdgeInsets.all(12.0),
                child: Text('No results', style: TextStyle(fontSize: 14)),
              ),
            );
          }

          if (data is RetrieverResponse) {
            return Card(
              key: const ValueKey('retriever'),
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Query: ${data.query ?? ''}'),
                      Text('Count: ${data.count ?? 0}'),
                      const SizedBox(height: 12),
                      ...data.results.map((r) {
                        final title =
                            r.passage?.title ??
                            r.passage?.sectionHeading ??
                            'Passage';
                        final snippet = (r.passage?.text ?? '').replaceAll(
                          '\n',
                          ' ',
                        );
                        return Card(
                          elevation: 1,
                          child: Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  title,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  snippet.length > 400
                                      ? '${snippet.substring(0, 400)}...'
                                      : snippet,
                                  style: const TextStyle(fontSize: 13),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'Score: ${r.finalScore ?? r.crossScore ?? r.faissScore ?? ''}',
                                ),
                              ],
                            ),
                          ),
                        );
                      }).toList(),
                    ],
                  ),
                ),
              ),
            );
          }

          return Card(
            key: const ValueKey('data'),
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(12.0),
              child: SingleChildScrollView(
                child: Text(
                  data.toString(),
                  style: const TextStyle(fontSize: 14),
                ),
              ),
            ),
          );
        },
        loading: () => Container(
          key: const ValueKey('loading'),
          child: _skeletonizer(),
        ),
        error: (err, st) => Card(
          key: const ValueKey('error'),
          color: Colors.red.shade50,
          child: Padding(
            padding: const EdgeInsets.all(12.0),
            child: Text(
              err.toString(),
              style: const TextStyle(color: Colors.red),
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _submitQa() async {
    final q = _qaController.text.trim();
    if (q.isEmpty) return;
    final topK = int.tryParse(_qaTopKController.text) ?? 10;
    final minScore = double.tryParse(_qaMinScoreController.text) ?? 0.4;
    setState(() {
      _currentQaQuestion = q;
    });

    // trigger the notifier to fetch
    await ref
        .read(qaProvider.notifier)
        .fetchAnswer(
          q,
          topK: topK,
          minScore: minScore,
        );
  }

  Future<void> _submitVerifier() async {
    if (_verifierUseBody) {
      final txt = _verifierBodyController.text.trim();
      if (txt.isEmpty) return;
      try {
        final body = json.decode(txt) as Map<String, dynamic>;
        setState(() => _currentVerifierKey = 'local');
        await ref.read(verifierProvider.notifier).verifyWithBody(body);
      } catch (e) {
        // set provider error by calling notifier method would handle it; fallback:
        setState(() => _currentVerifierKey = 'local');
        await ref.read(verifierProvider.notifier).verifyWithBody({
          'error': e.toString(),
        });
      }
    } else {
      final claim = _verifierClaimController.text.trim();
      if (claim.isEmpty) return;
      setState(() => _currentVerifierKey = claim);
      await ref.read(verifierProvider.notifier).verify(claim);
    }
  }

  Future<void> _submitRetriever() async {
    if (_retrieverUseBody) {
      final txt = _retrieverBodyController.text.trim();
      if (txt.isEmpty) return;
      try {
        final body = json.decode(txt) as Map<String, dynamic>;
        setState(() => _currentRetrieverKey = 'local');
        await ref.read(retrieverProvider.notifier).searchWithBody(body);
      } catch (e) {
        setState(() => _currentRetrieverKey = 'local');
        await ref.read(retrieverProvider.notifier).searchWithBody({
          'error': e.toString(),
        });
      }
    } else {
      final q = _retrieverQueryController.text.trim();
      if (q.isEmpty) return;
      setState(() => _currentRetrieverKey = q);
      await ref.read(retrieverProvider.notifier).search(q);
    }
  }

  @override
  Widget build(BuildContext context) {
    final AsyncValue<dynamic> qaState = _currentQaQuestion == null
        ? const AsyncValue.data(null)
        : ref.watch(qaProvider);

    final AsyncValue<dynamic> verifierState = _currentVerifierKey == null
        ? const AsyncValue.data(null)
        : ref.watch(verifierProvider);

    final AsyncValue<dynamic> retrieverState = _currentRetrieverKey == null
        ? const AsyncValue.data(null)
        : ref.watch(retrieverProvider);
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Response Test'),
          bottom: const TabBar(
            tabs: [
              Tab(text: 'QA'),
              Tab(text: 'Verifier'),
              Tab(text: 'Retriever'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            // QA Tab
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: Column(
                children: [
                  TextField(
                    controller: _qaController,
                    decoration: const InputDecoration(labelText: 'Question'),
                    minLines: 1,
                    maxLines: 3,
                  ),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _qaTopKController,
                          decoration: const InputDecoration(
                            labelText: 'topK',
                          ),
                          keyboardType: TextInputType.number,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: TextField(
                          controller: _qaMinScoreController,
                          decoration: const InputDecoration(
                            labelText: 'minScore',
                          ),
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      ElevatedButton(
                        onPressed: _submitQa,
                        child: const Text('Submit'),
                      ),
                      const SizedBox(width: 12),
                      ElevatedButton(
                        onPressed: () => _qaController.clear(),
                        child: const Text('Clear'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Expanded(child: _resultCard(qaState)),
                ],
              ),
            ),

            // Verifier Tab
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: Column(
                children: [
                  SwitchListTile(
                    title: const Text('Use JSON Body'),
                    value: _verifierUseBody,
                    onChanged: (v) => setState(() => _verifierUseBody = v),
                  ),
                  if (!_verifierUseBody) ...[
                    TextField(
                      controller: _verifierClaimController,
                      decoration: const InputDecoration(labelText: 'Claim'),
                      minLines: 1,
                      maxLines: 3,
                    ),
                  ] else ...[
                    TextField(
                      controller: _verifierBodyController,
                      decoration: const InputDecoration(
                        labelText: 'JSON Body',
                      ),
                      minLines: 3,
                      maxLines: 8,
                    ),
                  ],
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      ElevatedButton(
                        onPressed: _submitVerifier,
                        child: const Text('Submit'),
                      ),
                      const SizedBox(width: 12),
                      ElevatedButton(
                        onPressed: () {
                          _verifierClaimController.clear();
                          _verifierBodyController.clear();
                        },
                        child: const Text('Clear'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Expanded(child: _resultCard(verifierState)),
                ],
              ),
            ),

            // Retriever Tab
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: Column(
                children: [
                  SwitchListTile(
                    title: const Text('Use JSON Body'),
                    value: _retrieverUseBody,
                    onChanged: (v) => setState(() => _retrieverUseBody = v),
                  ),
                  if (!_retrieverUseBody) ...[
                    TextField(
                      controller: _retrieverQueryController,
                      decoration: const InputDecoration(labelText: 'Query'),
                      minLines: 1,
                      maxLines: 3,
                    ),
                  ] else ...[
                    TextField(
                      controller: _retrieverBodyController,
                      decoration: const InputDecoration(
                        labelText: 'JSON Body',
                      ),
                      minLines: 3,
                      maxLines: 8,
                    ),
                  ],
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      ElevatedButton(
                        onPressed: _submitRetriever,
                        child: const Text('Submit'),
                      ),
                      const SizedBox(width: 12),
                      ElevatedButton(
                        onPressed: () {
                          _retrieverQueryController.clear();
                          _retrieverBodyController.clear();
                        },
                        child: const Text('Clear'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Expanded(child: _resultCard(retrieverState)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

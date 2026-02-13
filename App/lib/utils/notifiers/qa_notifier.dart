import 'dart:async';

import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/models/qa_model.dart';
import 'package:verifact_app/utils/repositories/qa_repository.dart';

part 'qa_notifier.g.dart';

@riverpod
class QaNotifier extends _$QaNotifier {
  @override
  FutureOr<QAResponse> build(
    String question, {
    int topK = 10,
    double minScore = 0.4,
  }) {
    return ref
        .read(qaRepositoryProvider)
        .fetchAnswer(question, topK: topK, minScore: minScore);
  }

  Future<void> fetchAnswer(
    String question, {
    int topK = 10,
    double minScore = 0.4,
  }) async {
    state = AsyncValue<QAResponse>.loading();
    state = await AsyncValue.guard<QAResponse>(
      () => ref
          .read(qaRepositoryProvider)
          .fetchAnswer(question, topK: topK, minScore: minScore),
    );
  }
}

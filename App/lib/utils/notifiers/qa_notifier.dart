import 'dart:async';

import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/models/qa_model.dart';
import 'package:verifact_app/utils/repositories/qa_repository.dart';

part 'qa_notifier.g.dart';

@riverpod
class QaNotifier extends _$QaNotifier {
  @override
  FutureOr<QAModel?> build() {
    return null;
  }

  Future<void> fetchAnswer(
    String question, {
    int topK = 10,
    double minScore = 0.4,
  }) async {
    if (!ref.mounted) return;
    state = const AsyncValue.loading();

    final result = await AsyncValue.guard<QAModel>(
      () => ref
          .read(qaRepositoryProvider)
          .fetchAnswer(question, topK: topK, minScore: minScore),
    );

    if (!ref.mounted) return;
    state = result;
  }
}

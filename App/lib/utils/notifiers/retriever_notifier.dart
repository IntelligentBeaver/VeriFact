import 'dart:async';

import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/models/retriever_model.dart';
import 'package:verifact_app/utils/repositories/retriever_repository.dart';

part 'retriever_notifier.g.dart';

@riverpod
class RetrieverNotifier extends _$RetrieverNotifier {
  @override
  FutureOr<RetrieverResponse?> build() {
    return null;
  }

  Future<void> search(String query, {int topK = 10, double? minScore}) async {
    if (!ref.mounted) return;
    state = const AsyncValue.loading();

    final result = await AsyncValue.guard<RetrieverResponse>(
      () => ref
          .read(retrieverRepositoryProvider)
          .search(
            query,
            topK: topK,
            minScore: minScore,
          ),
    );

    if (!ref.mounted) return;
    state = result;
  }

  Future<void> searchWithBody(Map<String, dynamic> body) async {
    if (!ref.mounted) return;
    state = const AsyncValue.loading();

    final result = await AsyncValue.guard(
      () => ref.read(retrieverRepositoryProvider).searchWithBody(body),
    );

    if (!ref.mounted) return;
    state = result;
  }
}

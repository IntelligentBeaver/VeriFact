import 'dart:async';

import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/models/retriever_model.dart';
import 'package:verifact_app/utils/repositories/retriever_repository.dart';

part 'retriever_notifier.g.dart';

@riverpod
class RetrieverNotifier extends _$RetrieverNotifier {
  @override
  FutureOr<RetrieverResponse> build(
    String query, {
    int topK = 10,
    double? minScore,
  }) {
    return ref
        .read(retrieverRepositoryProvider)
        .search(query, topK: topK, minScore: minScore);
  }

  Future<void> search(String query, {int topK = 10, double? minScore}) async {
    state = AsyncValue<RetrieverResponse>.loading();
    state = await AsyncValue.guard<RetrieverResponse>(
      () => ref
          .read(retrieverRepositoryProvider)
          .search(query, topK: topK, minScore: minScore),
    );
  }

  Future<void> searchWithBody(Map<String, dynamic> body) async {
    state = AsyncValue<RetrieverResponse>.loading();
    state = await AsyncValue.guard<RetrieverResponse>(
      () => ref.read(retrieverRepositoryProvider).searchWithBody(body),
    );
  }
}

import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/controllers/retriever_controller.dart';
import 'package:verifact_app/models/retriever_model.dart';

part 'retriever_repository.g.dart';

/// Repository wrapper around `RetrieverController`.
///
/// Designed for use with Riverpod v3 code generation. Use the
/// generated provider `retrieverRepositoryProvider` to obtain an instance.
class RetrieverRepository {
  final RetrieverController _controller;

  RetrieverRepository([RetrieverController? controller])
    : _controller = controller ?? RetrieverController.instance;

  Future<RetrieverResponse> search(
    String query, {
    int topK = 10,
    double? minScore,
  }) {
    return _controller.search(query, topK: topK, minScore: minScore);
  }

  Future<RetrieverResponse> searchWithBody(Map<String, dynamic> body) {
    return _controller.searchWithBody(body);
  }
}

@riverpod
RetrieverRepository retrieverRepository(Ref ref) => RetrieverRepository();

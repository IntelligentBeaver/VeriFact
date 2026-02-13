import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/controllers/qa_controller.dart';
import 'package:verifact_app/models/qa_model.dart';

part 'qa_repository.g.dart';

/// Repository wrapper around `QAController`.
/// Use the generated provider `qaRepositoryProvider` to access an instance.
class QARepository {
  QARepository([QAController? controller])
    : _controller = controller ?? QAController.instance;
  final QAController _controller;

  Future<QAResponse> fetchAnswer(
    String question, {
    int topK = 10,
    double minScore = 0.4,
  }) {
    return _controller.fetchAnswer(question, topK: topK, minScore: minScore);
  }
}

@riverpod
QARepository qaRepository(Ref ref) => QARepository();

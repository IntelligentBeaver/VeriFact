import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/controllers/verifier_controller.dart';
import 'package:verifact_app/models/verifier_model.dart';

part 'verifier_repository.g.dart';

/// Repository wrapper around `VerifierController`.
/// Use the generated provider `verifierRepositoryProvider` to access an instance.
class VerifierRepository {

  VerifierRepository([VerifierController? controller])
    : _controller = controller ?? VerifierController.instance;
  final VerifierController _controller;

  Future<VerifierModel> verifyClaim(String claim) {
    return _controller.verifyClaim(claim);
  }

  Future<VerifierModel> verifyClaimWithBody(Map<String, dynamic> body) {
    return _controller.verifyClaimWithBody(body);
  }
}

@riverpod
VerifierRepository verifierRepository(Ref ref) => VerifierRepository();

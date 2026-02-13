import 'dart:async';

import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/models/verifier_model.dart';
import 'package:verifact_app/utils/repositories/verifier_repository.dart';

part 'verifier_notifier.g.dart';

@riverpod
class VerifierNotifier extends _$VerifierNotifier {
  @override
  FutureOr<VerifierModel> build(String claim) {
    return ref.read(verifierRepositoryProvider).verifyClaim(claim);
  }

  Future<void> verify(String claim) async {
    state = const AsyncValue<VerifierModel>.loading();
    state = await AsyncValue.guard<VerifierModel>(
      () => ref.read(verifierRepositoryProvider).verifyClaim(claim),
    );
  }

  Future<void> verifyWithBody(Map<String, dynamic> body) async {
    state = const AsyncValue<VerifierModel>.loading();
    state = await AsyncValue.guard<VerifierModel>(
      () => ref.read(verifierRepositoryProvider).verifyClaimWithBody(body),
    );
  }
}

import 'dart:async';

import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:verifact_app/models/verifier_model.dart';
import 'package:verifact_app/utils/repositories/verifier_repository.dart';

part 'verifier_notifier.g.dart';

@riverpod
class VerifierNotifier extends _$VerifierNotifier {
  @override
  FutureOr<VerifierModel?> build() {
    return null;
  }

  Future<void> verify(String claim) async {
    if (!ref.mounted) return;
    state = const AsyncValue<VerifierModel>.loading();

    final result = await AsyncValue.guard<VerifierModel>(
      () => ref.read(verifierRepositoryProvider).verifyClaim(claim),
    );

    if (!ref.mounted) return;
    state = result;
  }

  Future<void> verifyWithBody(Map<String, dynamic> body) async {
    if (!ref.mounted) return;
    state = const AsyncValue<VerifierModel>.loading();

    final result = await AsyncValue.guard<VerifierModel>(
      () => ref.read(verifierRepositoryProvider).verifyClaimWithBody(body),
    );

    if (!ref.mounted) return;
    state = result;
  }
}

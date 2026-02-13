// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'verifier_notifier.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(VerifierNotifier)
final verifierProvider = VerifierNotifierProvider._();

final class VerifierNotifierProvider
    extends $AsyncNotifierProvider<VerifierNotifier, VerifierModel?> {
  VerifierNotifierProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'verifierProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$verifierNotifierHash();

  @$internal
  @override
  VerifierNotifier create() => VerifierNotifier();
}

String _$verifierNotifierHash() => r'f4200013cffbb619544d1f4f6fdd04b7881d7d50';

abstract class _$VerifierNotifier extends $AsyncNotifier<VerifierModel?> {
  FutureOr<VerifierModel?> build();
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<AsyncValue<VerifierModel?>, VerifierModel?>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<AsyncValue<VerifierModel?>, VerifierModel?>,
              AsyncValue<VerifierModel?>,
              Object?,
              Object?
            >;
    element.handleCreate(ref, build);
  }
}

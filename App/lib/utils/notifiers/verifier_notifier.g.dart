// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'verifier_notifier.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(VerifierNotifier)
final verifierProvider = VerifierNotifierFamily._();

final class VerifierNotifierProvider
    extends $AsyncNotifierProvider<VerifierNotifier, VerifierModel> {
  VerifierNotifierProvider._({
    required VerifierNotifierFamily super.from,
    required String super.argument,
  }) : super(
         retry: null,
         name: r'verifierProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$verifierNotifierHash();

  @override
  String toString() {
    return r'verifierProvider'
        ''
        '($argument)';
  }

  @$internal
  @override
  VerifierNotifier create() => VerifierNotifier();

  @override
  bool operator ==(Object other) {
    return other is VerifierNotifierProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$verifierNotifierHash() => r'76cda0b8db797487ac4dd367b3d1976dbdaf1b5f';

final class VerifierNotifierFamily extends $Family
    with
        $ClassFamilyOverride<
          VerifierNotifier,
          AsyncValue<VerifierModel>,
          VerifierModel,
          FutureOr<VerifierModel>,
          String
        > {
  VerifierNotifierFamily._()
    : super(
        retry: null,
        name: r'verifierProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  VerifierNotifierProvider call(String claim) =>
      VerifierNotifierProvider._(argument: claim, from: this);

  @override
  String toString() => r'verifierProvider';
}

abstract class _$VerifierNotifier extends $AsyncNotifier<VerifierModel> {
  late final _$args = ref.$arg as String;
  String get claim => _$args;

  FutureOr<VerifierModel> build(String claim);
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<AsyncValue<VerifierModel>, VerifierModel>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<AsyncValue<VerifierModel>, VerifierModel>,
              AsyncValue<VerifierModel>,
              Object?,
              Object?
            >;
    element.handleCreate(ref, () => build(_$args));
  }
}

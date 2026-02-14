// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'verifier_repository.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(verifierRepository)
final verifierRepositoryProvider = VerifierRepositoryProvider._();

final class VerifierRepositoryProvider
    extends
        $FunctionalProvider<
          VerifierRepository,
          VerifierRepository,
          VerifierRepository
        >
    with $Provider<VerifierRepository> {
  VerifierRepositoryProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'verifierRepositoryProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$verifierRepositoryHash();

  @$internal
  @override
  $ProviderElement<VerifierRepository> $createElement(
    $ProviderPointer pointer,
  ) => $ProviderElement(pointer);

  @override
  VerifierRepository create(Ref ref) {
    return verifierRepository(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(VerifierRepository value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<VerifierRepository>(value),
    );
  }
}

String _$verifierRepositoryHash() =>
    r'f0ce3f2d7f0dc2d60321adae4013354746034ee7';

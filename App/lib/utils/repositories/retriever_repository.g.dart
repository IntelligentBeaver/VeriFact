// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'retriever_repository.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(retrieverRepository)
final retrieverRepositoryProvider = RetrieverRepositoryProvider._();

final class RetrieverRepositoryProvider
    extends
        $FunctionalProvider<
          RetrieverRepository,
          RetrieverRepository,
          RetrieverRepository
        >
    with $Provider<RetrieverRepository> {
  RetrieverRepositoryProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'retrieverRepositoryProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$retrieverRepositoryHash();

  @$internal
  @override
  $ProviderElement<RetrieverRepository> $createElement(
    $ProviderPointer pointer,
  ) => $ProviderElement(pointer);

  @override
  RetrieverRepository create(Ref ref) {
    return retrieverRepository(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(RetrieverRepository value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<RetrieverRepository>(value),
    );
  }
}

String _$retrieverRepositoryHash() =>
    r'1dff8d07742983a7944f6bf4d25ace25013faf3c';

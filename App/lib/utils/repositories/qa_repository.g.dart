// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'qa_repository.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(qaRepository)
final qaRepositoryProvider = QaRepositoryProvider._();

final class QaRepositoryProvider
    extends $FunctionalProvider<QARepository, QARepository, QARepository>
    with $Provider<QARepository> {
  QaRepositoryProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'qaRepositoryProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$qaRepositoryHash();

  @$internal
  @override
  $ProviderElement<QARepository> $createElement($ProviderPointer pointer) =>
      $ProviderElement(pointer);

  @override
  QARepository create(Ref ref) {
    return qaRepository(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(QARepository value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<QARepository>(value),
    );
  }
}

String _$qaRepositoryHash() => r'9985894ebfa12c74faf200ee0d87f37f787dd71a';

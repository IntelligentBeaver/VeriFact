// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'retriever_notifier.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(RetrieverNotifier)
final retrieverProvider = RetrieverNotifierProvider._();

final class RetrieverNotifierProvider
    extends $AsyncNotifierProvider<RetrieverNotifier, RetrieverResponse?> {
  RetrieverNotifierProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'retrieverProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$retrieverNotifierHash();

  @$internal
  @override
  RetrieverNotifier create() => RetrieverNotifier();
}

String _$retrieverNotifierHash() => r'00902a93e1dafb2f534f47ae5dcb57a7ceff0e2b';

abstract class _$RetrieverNotifier extends $AsyncNotifier<RetrieverResponse?> {
  FutureOr<RetrieverResponse?> build();
  @$mustCallSuper
  @override
  void runBuild() {
    final ref =
        this.ref as $Ref<AsyncValue<RetrieverResponse?>, RetrieverResponse?>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<AsyncValue<RetrieverResponse?>, RetrieverResponse?>,
              AsyncValue<RetrieverResponse?>,
              Object?,
              Object?
            >;
    element.handleCreate(ref, build);
  }
}

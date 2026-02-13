// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'retriever_notifier.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(RetrieverNotifier)
final retrieverProvider = RetrieverNotifierFamily._();

final class RetrieverNotifierProvider
    extends $AsyncNotifierProvider<RetrieverNotifier, RetrieverResponse> {
  RetrieverNotifierProvider._({
    required RetrieverNotifierFamily super.from,
    required (String, {int topK, double? minScore}) super.argument,
  }) : super(
         retry: null,
         name: r'retrieverProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$retrieverNotifierHash();

  @override
  String toString() {
    return r'retrieverProvider'
        ''
        '$argument';
  }

  @$internal
  @override
  RetrieverNotifier create() => RetrieverNotifier();

  @override
  bool operator ==(Object other) {
    return other is RetrieverNotifierProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$retrieverNotifierHash() => r'92df9b95df842f0d1e7eaace627496c306c61d7e';

final class RetrieverNotifierFamily extends $Family
    with
        $ClassFamilyOverride<
          RetrieverNotifier,
          AsyncValue<RetrieverResponse>,
          RetrieverResponse,
          FutureOr<RetrieverResponse>,
          (String, {int topK, double? minScore})
        > {
  RetrieverNotifierFamily._()
    : super(
        retry: null,
        name: r'retrieverProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  RetrieverNotifierProvider call(
    String query, {
    int topK = 10,
    double? minScore,
  }) => RetrieverNotifierProvider._(
    argument: (query, topK: topK, minScore: minScore),
    from: this,
  );

  @override
  String toString() => r'retrieverProvider';
}

abstract class _$RetrieverNotifier extends $AsyncNotifier<RetrieverResponse> {
  late final _$args = ref.$arg as (String, {int topK, double? minScore});
  String get query => _$args.$1;
  int get topK => _$args.topK;
  double? get minScore => _$args.minScore;

  FutureOr<RetrieverResponse> build(
    String query, {
    int topK = 10,
    double? minScore,
  });
  @$mustCallSuper
  @override
  void runBuild() {
    final ref =
        this.ref as $Ref<AsyncValue<RetrieverResponse>, RetrieverResponse>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<AsyncValue<RetrieverResponse>, RetrieverResponse>,
              AsyncValue<RetrieverResponse>,
              Object?,
              Object?
            >;
    element.handleCreate(
      ref,
      () => build(_$args.$1, topK: _$args.topK, minScore: _$args.minScore),
    );
  }
}

// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'qa_notifier.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(QaNotifier)
final qaProvider = QaNotifierFamily._();

final class QaNotifierProvider
    extends $AsyncNotifierProvider<QaNotifier, QAResponse> {
  QaNotifierProvider._({
    required QaNotifierFamily super.from,
    required (String, {int topK, double minScore}) super.argument,
  }) : super(
         retry: null,
         name: r'qaProvider',
         isAutoDispose: true,
         dependencies: null,
         $allTransitiveDependencies: null,
       );

  @override
  String debugGetCreateSourceHash() => _$qaNotifierHash();

  @override
  String toString() {
    return r'qaProvider'
        ''
        '$argument';
  }

  @$internal
  @override
  QaNotifier create() => QaNotifier();

  @override
  bool operator ==(Object other) {
    return other is QaNotifierProvider && other.argument == argument;
  }

  @override
  int get hashCode {
    return argument.hashCode;
  }
}

String _$qaNotifierHash() => r'a1513fa6881a6e74dd3f95dfcb140abc9bd8c62e';

final class QaNotifierFamily extends $Family
    with
        $ClassFamilyOverride<
          QaNotifier,
          AsyncValue<QAResponse>,
          QAResponse,
          FutureOr<QAResponse>,
          (String, {int topK, double minScore})
        > {
  QaNotifierFamily._()
    : super(
        retry: null,
        name: r'qaProvider',
        dependencies: null,
        $allTransitiveDependencies: null,
        isAutoDispose: true,
      );

  QaNotifierProvider call(
    String question, {
    int topK = 10,
    double minScore = 0.4,
  }) => QaNotifierProvider._(
    argument: (question, topK: topK, minScore: minScore),
    from: this,
  );

  @override
  String toString() => r'qaProvider';
}

abstract class _$QaNotifier extends $AsyncNotifier<QAResponse> {
  late final _$args = ref.$arg as (String, {int topK, double minScore});
  String get question => _$args.$1;
  int get topK => _$args.topK;
  double get minScore => _$args.minScore;

  FutureOr<QAResponse> build(
    String question, {
    int topK = 10,
    double minScore = 0.4,
  });
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<AsyncValue<QAResponse>, QAResponse>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<AsyncValue<QAResponse>, QAResponse>,
              AsyncValue<QAResponse>,
              Object?,
              Object?
            >;
    element.handleCreate(
      ref,
      () => build(_$args.$1, topK: _$args.topK, minScore: _$args.minScore),
    );
  }
}

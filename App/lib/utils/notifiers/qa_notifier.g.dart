// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'qa_notifier.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(QaNotifier)
final qaProvider = QaNotifierProvider._();

final class QaNotifierProvider
    extends $AsyncNotifierProvider<QaNotifier, QAModel?> {
  QaNotifierProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'qaProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$qaNotifierHash();

  @$internal
  @override
  QaNotifier create() => QaNotifier();
}

String _$qaNotifierHash() => r'95fd8c12a2b2b6350a898b5a8cedb43014fc0afc';

abstract class _$QaNotifier extends $AsyncNotifier<QAModel?> {
  FutureOr<QAModel?> build();
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<AsyncValue<QAModel?>, QAModel?>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<AsyncValue<QAModel?>, QAModel?>,
              AsyncValue<QAModel?>,
              Object?,
              Object?
            >;
    element.handleCreate(ref, build);
  }
}

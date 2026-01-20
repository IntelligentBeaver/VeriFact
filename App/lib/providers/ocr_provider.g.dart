// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'ocr_provider.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning
/// OCR Service Provider

@ProviderFor(ocrService)
final ocrServiceProvider = OcrServiceProvider._();

/// OCR Service Provider

final class OcrServiceProvider
    extends $FunctionalProvider<OcrService, OcrService, OcrService>
    with $Provider<OcrService> {
  /// OCR Service Provider
  OcrServiceProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'ocrServiceProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$ocrServiceHash();

  @$internal
  @override
  $ProviderElement<OcrService> $createElement($ProviderPointer pointer) =>
      $ProviderElement(pointer);

  @override
  OcrService create(Ref ref) {
    return ocrService(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(OcrService value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<OcrService>(value),
    );
  }
}

String _$ocrServiceHash() => r'0ad770d51dba1ddbf251db06309ce4968a7c731a';

/// OCR State Notifier Provider

@ProviderFor(OcrNotifier)
final ocrProvider = OcrNotifierProvider._();

/// OCR State Notifier Provider
final class OcrNotifierProvider
    extends $NotifierProvider<OcrNotifier, OcrState> {
  /// OCR State Notifier Provider
  OcrNotifierProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'ocrProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$ocrNotifierHash();

  @$internal
  @override
  OcrNotifier create() => OcrNotifier();

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(OcrState value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<OcrState>(value),
    );
  }
}

String _$ocrNotifierHash() => r'f697d651fadc60b5d9008144c1a44f04ca3d2602';

/// OCR State Notifier Provider

abstract class _$OcrNotifier extends $Notifier<OcrState> {
  OcrState build();
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<OcrState, OcrState>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<OcrState, OcrState>,
              OcrState,
              Object?,
              Object?
            >;
    element.handleCreate(ref, build);
  }
}

/// Combined OCR state provider for easy access in widgets

@ProviderFor(ocrState)
final ocrStateProvider = OcrStateProvider._();

/// Combined OCR state provider for easy access in widgets

final class OcrStateProvider
    extends $FunctionalProvider<OcrState, OcrState, OcrState>
    with $Provider<OcrState> {
  /// Combined OCR state provider for easy access in widgets
  OcrStateProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'ocrStateProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$ocrStateHash();

  @$internal
  @override
  $ProviderElement<OcrState> $createElement($ProviderPointer pointer) =>
      $ProviderElement(pointer);

  @override
  OcrState create(Ref ref) {
    return ocrState(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(OcrState value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<OcrState>(value),
    );
  }
}

String _$ocrStateHash() => r'7e93ca36317e66798b75650434ded73c65042335';

/// Check if OCR is currently processing

@ProviderFor(isOcrProcessing)
final isOcrProcessingProvider = IsOcrProcessingProvider._();

/// Check if OCR is currently processing

final class IsOcrProcessingProvider
    extends $FunctionalProvider<bool, bool, bool>
    with $Provider<bool> {
  /// Check if OCR is currently processing
  IsOcrProcessingProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'isOcrProcessingProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$isOcrProcessingHash();

  @$internal
  @override
  $ProviderElement<bool> $createElement($ProviderPointer pointer) =>
      $ProviderElement(pointer);

  @override
  bool create(Ref ref) {
    return isOcrProcessing(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(bool value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<bool>(value),
    );
  }
}

String _$isOcrProcessingHash() => r'653816362b8c4eb3e467839a6588b674b4ae6717';

/// Check if text has been recognized

@ProviderFor(hasRecognizedText)
final hasRecognizedTextProvider = HasRecognizedTextProvider._();

/// Check if text has been recognized

final class HasRecognizedTextProvider
    extends $FunctionalProvider<bool, bool, bool>
    with $Provider<bool> {
  /// Check if text has been recognized
  HasRecognizedTextProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'hasRecognizedTextProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$hasRecognizedTextHash();

  @$internal
  @override
  $ProviderElement<bool> $createElement($ProviderPointer pointer) =>
      $ProviderElement(pointer);

  @override
  bool create(Ref ref) {
    return hasRecognizedText(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(bool value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<bool>(value),
    );
  }
}

String _$hasRecognizedTextHash() => r'90cbc230b3c7bdc33208a5cd905494dfd3dfe77d';

/// Get recognized text

@ProviderFor(recognizedText)
final recognizedTextProvider = RecognizedTextProvider._();

/// Get recognized text

final class RecognizedTextProvider
    extends $FunctionalProvider<String, String, String>
    with $Provider<String> {
  /// Get recognized text
  RecognizedTextProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'recognizedTextProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$recognizedTextHash();

  @$internal
  @override
  $ProviderElement<String> $createElement($ProviderPointer pointer) =>
      $ProviderElement(pointer);

  @override
  String create(Ref ref) {
    return recognizedText(ref);
  }

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(String value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<String>(value),
    );
  }
}

String _$recognizedTextHash() => r'eb732cf55bc04977a9d874bf6bc11a69929feff3';

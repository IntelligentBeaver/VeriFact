// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'home_search_notifier.dart';

// **************************************************************************
// RiverpodGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint, type=warning

@ProviderFor(HomeSearchNotifier)
final homeSearchProvider = HomeSearchNotifierProvider._();

final class HomeSearchNotifierProvider
    extends $NotifierProvider<HomeSearchNotifier, HomeSearchMode> {
  HomeSearchNotifierProvider._()
    : super(
        from: null,
        argument: null,
        retry: null,
        name: r'homeSearchProvider',
        isAutoDispose: true,
        dependencies: null,
        $allTransitiveDependencies: null,
      );

  @override
  String debugGetCreateSourceHash() => _$homeSearchNotifierHash();

  @$internal
  @override
  HomeSearchNotifier create() => HomeSearchNotifier();

  /// {@macro riverpod.override_with_value}
  Override overrideWithValue(HomeSearchMode value) {
    return $ProviderOverride(
      origin: this,
      providerOverride: $SyncValueProvider<HomeSearchMode>(value),
    );
  }
}

String _$homeSearchNotifierHash() =>
    r'dd74811b7fba37188ba48e37f8056ef8b172d95d';

abstract class _$HomeSearchNotifier extends $Notifier<HomeSearchMode> {
  HomeSearchMode build();
  @$mustCallSuper
  @override
  void runBuild() {
    final ref = this.ref as $Ref<HomeSearchMode, HomeSearchMode>;
    final element =
        ref.element
            as $ClassProviderElement<
              AnyNotifier<HomeSearchMode, HomeSearchMode>,
              HomeSearchMode,
              Object?,
              Object?
            >;
    element.handleCreate(ref, build);
  }
}

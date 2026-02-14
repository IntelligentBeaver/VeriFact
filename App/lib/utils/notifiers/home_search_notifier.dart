import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'home_search_notifier.g.dart';

enum HomeSearchMode { verifier, qa, doc }

@riverpod
class HomeSearchNotifier extends _$HomeSearchNotifier {
  @override
  HomeSearchMode build() => HomeSearchMode.verifier;

  void selectMode(HomeSearchMode mode) {
    state = mode;
  }
}

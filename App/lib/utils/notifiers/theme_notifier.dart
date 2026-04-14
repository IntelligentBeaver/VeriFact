import 'dart:async';

import 'package:flutter/material.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:verifact_app/utils/helpers/helper_functions.dart';

part 'theme_notifier.g.dart';

@riverpod
class ThemeNotifier extends _$ThemeNotifier {
  static const String _prefKey = 'theme_mode';

  @override
  FutureOr<ThemeMode> build() async {
    final prefs = await SharedPreferences.getInstance();
    final stored = prefs.getString(_prefKey) ?? 'light';
    return mapToThemeMode(stored);
  }

  Future<void> setTheme(ThemeMode mode) async {
    state = AsyncValue.data(mode);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKey, _themeModeToString(mode));
  }

  Future<void> toggleTheme() async {
    final current = state.value ?? ThemeMode.light;
    final next = current == ThemeMode.light ? ThemeMode.dark : ThemeMode.light;
    await setTheme(next);
  }

  String _themeModeToString(ThemeMode mode) {
    switch (mode) {
      case ThemeMode.dark:
        return 'dark';
      case ThemeMode.light:
      case ThemeMode.system:
        return 'light';
    }
  }
}

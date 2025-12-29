import 'dart:ui';

import 'package:shared_preferences/shared_preferences.dart';

Future<void> saveLocaleToSharedPreference(Locale newLocale) async {
  final prefs = await SharedPreferences.getInstance();
  await prefs.setString('selected_language', newLocale.languageCode);
  await prefs.setString('selected_country', newLocale.countryCode ?? '');
}

Future<Locale> loadLocaleFromSharedPreference() async {
  final prefs = await SharedPreferences.getInstance();
  final languageCode = prefs.getString('selected_language');
  final countryCode = prefs.getString('selected_country');
  var locale = const Locale('en', 'US');
  if (languageCode != null) {
    locale = Locale(languageCode, countryCode);
  }
  return locale;
}

Future<void> clearSharedPreferences() async {
  final prefs = await SharedPreferences.getInstance();
  await prefs.clear();
}

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _kLocalePrefKey = 'app_locale';

const kUserJson = 'user_json';
const kOAuthAccountJson = 'oauth_account_json';
const kIsFirstTimeOAuth = 'is_first_time_oauth';

// Cached instance to avoid repeated getInstance() calls
SharedPreferences? _cachedPrefs;

Future<SharedPreferences> _getPrefs() async {
  _cachedPrefs ??= await SharedPreferences.getInstance();
  return _cachedPrefs!;
}

Future<void> setLocalePreference(String locale) async {
  final prefs = await _getPrefs();
  await prefs.setString(_kLocalePrefKey, locale);
}

Future<String?> getLocalePreference() async {
  final prefs = await _getPrefs();
  return prefs.getString(_kLocalePrefKey);
}

/// Function to set token in Shared Preferences
Future<bool> setAuthTokenPreference({required String token}) async {
  // Save token
  final prefs = await _getPrefs();
  await prefs.setString('auth_token', token);
  return true;
}

/// Function to get token from Shared Preferences
Future<String?> getAuthTokenPreference() async {
  // Retrieve token
  final prefs = await _getPrefs();
  return prefs.getString('auth_token');
}

/// Function to clear token from Shared Preferences
Future<void> clearAuthTokenPreference() async {
  final prefs = await _getPrefs();
  await prefs.remove('auth_token');
}

/// Function to set User details in Shared Preferences
Future<void> setUserDetailsPreference({required String userJson}) async {
  final prefs = await _getPrefs();
  await prefs.setString(kUserJson, userJson);
}

/// Function to get User details from Shared Preferences
Future<String?> getUserDetailsPreference() async {
  final prefs = await _getPrefs();
  return prefs.getString(kUserJson);
}

/// Function to clear User details from Shared Preferences
Future<void> clearUserDetailsPreference() async {
  final prefs = await _getPrefs();
  await prefs.remove(kUserJson);
}

/// Function to know if this is first time opening of app
Future<bool> getIsFirstTimeOpen() async {
  final prefs = await _getPrefs();
  final isFirstTime = prefs.getBool('isFirstTime') ?? true;
  return isFirstTime;
}

/// After opening, this is set to know this is not first time of app opening
Future<void> setIsFirstTimeOpen({required bool value}) async {
  final prefs = await _getPrefs();
  await prefs.setBool('isFirstTime', value);
}

/// Function to know if this is first time opening of app
Future<bool> isFirstTimeOpen() async {
  final prefs = await _getPrefs();
  final isFirstTime = prefs.getBool('isFirstTime') ?? true;
  return isFirstTime;
}

/// Function to get the [ThemeData] from Shared Preferences
Future<String?> getThemePreference() async {
  final prefs = await _getPrefs();
  final mode = prefs.getString('theme_mode');
  return mode;
}

/// Function to set the [ThemeData] in Shared Preferences
Future<void> setThemePreference({required String mode}) async {
  final prefs = await _getPrefs();
  await prefs.setString('theme_mode', mode);
}

/// Function to clear the [ThemeData] from Shared Preferences
Future<void> clearThemePreference() async {
  final prefs = await SharedPreferences.getInstance();
  await prefs.remove('theme_mode');
}

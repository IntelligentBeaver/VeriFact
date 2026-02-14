import 'dart:convert';
import 'dart:io';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher_string.dart';
import 'package:verifact_app/utils/constants/app_globals.dart';
import 'package:verifact_app/widgets/custom_snackbar.dart';

/// Displays an error [SnackBar] with the given [message].
///
/// Optionally, you can specify the display [time] in milliseconds (default is 1000).
/// If [actionMethod] and [actionLabel] are provided, an action button will be shown
/// in the [SnackBar] that triggers [actionMethod] when pressed.
///
/// The [SnackBar] uses [AppColors.error] as its background color and applies
/// padding using [AppSizes.sm] and [AppSizes.md].
///
/// Removes any currently displayed [SnackBar] before showing the new one.
///
/// Example usage:
/// ```dart
/// showErrorSnackbar('An error occurred');
/// showErrorSnackbar(
///   'Failed to save',
///   actionLabel: 'Retry',
///   actionMethod: () { /* retry logic */ },
/// );
/// ```
/// Requires [navigatorKey] to be properly initialized and accessible.
void showErrorSnackbar(
  String message, {
  BuildContext? context,
}) {
  TopSnackbar.show(
    navigatorKey.currentContext!,
    TopSnackbar.error(message),
  );
}

/// Displays a success [SnackBar] with the given [message].
///
/// Optionally, you can specify the display [time] in milliseconds (default is 1000).
/// If [actionMethod] and [actionLabel] are provided, an action button will be shown in the snackbar.
///
/// Before showing the new snackbar, any currently displayed snackbar will be hidden.
///
/// Example usage:
/// ```dart
/// showSuccessSnackbar('Operation successful');
/// showSuccessSnackbar(
///   'Item deleted',
///   actionLabel: 'UNDO',
///   actionMethod: () { /* undo logic */ },
/// );
/// ```
///
/// Requires [navigatorKey] to be properly initialized and accessible.
void showSuccessSnackbar(
  String message, {
  BuildContext? context,
}) {
  TopSnackbar.show(
    navigatorKey.currentContext!,
    TopSnackbar.success(message),
  );
}

/// Displays an informational [SnackBar] with the given [message].
///
/// Optionally, you can specify the display [time] in milliseconds (default is 1000).
/// If [actionMethod] and [actionLabel] are provided, an action button will be shown
/// in the [SnackBar] that triggers [actionMethod] when pressed.
///
/// The function first hides any currently displayed [SnackBar] before showing the new one.
///
/// Example usage:
/// ```dart
/// showInfoSnackbar('This is an info message');
/// showInfoSnackbar(
///   'Undo action',
///   actionLabel: 'UNDO',
///   actionMethod: () { /* undo logic */ },
/// );
/// ```
/// Requires [navigatorKey] to be properly initialized and accessible.
void showInfoSnackbar(
  String message, {
  BuildContext? context,
  Duration duration = const Duration(milliseconds: 2000),
}) {
  TopSnackbar.show(
    duration: duration,
    navigatorKey.currentContext!,
    TopSnackbar.info(message),
  );
}

/// Retrieves a unique device identifier as a JSON-encoded string.
///
/// On Android devices, this includes:
/// - `serial_number`: The device's serial number.
/// - `manufacturer`: The device manufacturer (e.g., Samsung, Xiaomi).
/// - `model`: The device model (e.g., Galaxy S21).
/// - `version`: The Android OS version (release string).
///
/// On non-Android platforms, the returned string will default to `'{}'`.
///
/// Example usage:
/// ```dart
/// String deviceKey = await getDeviceKey();
/// print(deviceKey); // {"serial_number":"ABC123","manufacturer":"Samsung", ...}
/// ```
///
/// This method uses the `device_info_plus` package and requires appropriate
/// permissions to access device information on Android.
Future<String> getDeviceKey() async {
  final deviceInfoPlugin = DeviceInfoPlugin();
  var deviceKey = '{}';

  if (Platform.isAndroid) {
    final androidInfo = await deviceInfoPlugin.androidInfo;
    deviceKey = jsonEncode({
      'serial_number': androidInfo.id,
      'manufacturer': androidInfo.manufacturer,
      'model': androidInfo.model,
      'version': androidInfo.version.release,
    });
  } else if (Platform.isIOS) {
    final iosInfo = await deviceInfoPlugin.iosInfo;
    deviceKey = jsonEncode({
      'name': iosInfo.name,
      'systemName': iosInfo.systemName,
      'systemVersion': iosInfo.systemVersion,
      'model': iosInfo.model,
      'identifierForVendor': iosInfo.identifierForVendor,
    });
  }

  return deviceKey;
}

/// Checks if the device is currently connected to the internet.
///
/// Uses the `connectivity_plus` package to determine the current network status.
/// Returns `true` if the device is connected via Wi-Fi, mobile data, or any
/// connection other than `none` or `vpn`.
///
/// Example usage:
/// ```dart
/// bool online = await isConnected();
/// if (online) {
///   print('Device is online');
/// } else {
///   print('No internet connection');
/// }
/// ```
///
/// This function only checks for network availability and does not verify
/// actual internet access.
Future<bool> isConnected() async {
  final results = await Connectivity().checkConnectivity();

  return results.any(
    (result) =>
        result != ConnectivityResult.none && result != ConnectivityResult.vpn,
  );
}

/// Converts String to ThemeMode.
ThemeMode mapToThemeMode(String value) {
  switch (value) {
    case 'light':
      return ThemeMode.light;
    case 'dark':
      return ThemeMode.dark;
    default:
      return ThemeMode.light;
  }
}

/// Allows to open a URL by sepcifying the URL String.
Future<void> launchUrl(String url, {required mode}) async {
  if (!await launchUrlString(url, mode: LaunchMode.externalApplication)) {
    throw Exception('Could not launch $url');
  }
}

String parseDuration(Duration duration, {bool shorthand = false}) {
  if (duration.inHours >= 1) {
    final h = duration.inHours;
    if (shorthand) {
      return '$h ${h == 1 ? "hr" : "hrs"}';
    }
    return '$h ${h == 1 ? "hr" : "hrs"}';
  }

  if (duration.inMinutes >= 1) {
    final m = duration.inMinutes;
    if (shorthand) {
      return '$m ${m == 1 ? "min" : "mins"}';
    }
    return '$m ${m == 1 ? "minute" : "minutes"}';
  }

  final s = duration.inSeconds;
  if (shorthand) {
    return '$s ${s == 1 ? "sec" : "secs"}';
  }
  return '$s ${s == 1 ? "second" : "seconds"}';
}

String formatDuration(Duration d) {
  final hours = d.inHours.toString().padLeft(2, '0');
  final minutes = (d.inMinutes % 60).toString().padLeft(2, '0');
  final seconds = (d.inSeconds % 60).toString().padLeft(2, '0');

  return '$hours:$minutes:$seconds';
}

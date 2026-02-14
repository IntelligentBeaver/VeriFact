import 'dart:async';

import 'package:permission_handler/permission_handler.dart';

/// Requests all required permissions and services at application startup.
///
/// This function sequentially checks and requests the following:
/// - Notification permission
/// - Location permission
/// - Location service status
/// - Bluetooth permission
/// - Bluetooth connect permission
/// - Bluetooth service status
///
/// Ensures that the application has all necessary permissions and services enabled
/// for proper functionality at boot time.
Future<void> requestRequiredPermissionAtBoot() async {
  await checkNotificationPermission();
  // await checkLocationPermission();
  // await checkLocationService();
  // await checkBluetoothPermission();
  // await checkBluetoothConnectPermission();
  // await checkBluetoothService();
}

/// Checks if the app has notification permissions.
///
/// Returns `true` if the notification permission is already granted.
/// If not, it requests the permission and returns `true` if the user grants it,
/// or `false` otherwise.
///
/// Uses the `Permission.notification` from the `permission_handler` package.
Future<bool> checkNotificationPermission() async {
  if (await Permission.notification.isGranted) {
    return true;
  } else {
    final status = await Permission.notification.request();
    return status.isGranted;
  }
}

/// Checks and requests location permission from the user.
///
/// This function first checks if the location permission (`locationWhenInUse`)
/// has already been granted. If it has, the function returns `true`.
/// If not, it requests the permission from the user and returns `true` if
/// the permission is granted after the request, or `false` otherwise.
///
/// Returns:
///   - `true` if the location permission is granted.
///   - `false` if the location permission is denied.
// Future<bool> checkLocationPermission() async {
//   if (await Permission.locationWhenInUse.isGranted) {
//     return true;
//   } else {
//     final status = await Permission.locationWhenInUse.request();
//     return status.isGranted;
//   }
// }

/// Checks if the device's location service is enabled.
///
/// If the location service is enabled, returns `true`.
/// If not, attempts to request the user to enable the location service.
/// If the user enables the service, returns `true`.
/// Otherwise, displays an informational snackbar prompting the user to enable
/// location services via app settings, and returns `false`.
///
/// Returns:
///   - `true` if location service is enabled or successfully enabled by the user.
///   - `false` if the service remains disabled after prompting the user.
// Future<bool> checkLocationService() async {
//   final isLocationEnabled = await Permission.location.serviceStatus.isEnabled;
//   if (isLocationEnabled) {
//     return true;
//   } else {
//     final location = Location();
//     if (await location.requestService()) {
//       return true;
//     }
//     showInfoSnackbar(
//       'Location services are disabled. Please enable Location.',
//     );
//     return false;
//   }
// }

/// Checks if the Bluetooth permission is granted.
///
/// If the permission is already granted, returns `true`.
/// Otherwise, requests the Bluetooth permission from the user and
/// returns `true` if the permission is granted after the request,
/// or `false` if it is denied.
///
/// Returns:
///   - `Future<bool>`: A future that resolves to `true` if the Bluetooth
///     permission is granted, otherwise `false`.
// Future<bool> checkBluetoothPermission() async {
//   if (await Permission.bluetooth.isGranted) {
//     return true;
//   } else {
//     final status = await Permission.bluetooth.request();
//     return status.isGranted;
//   }
// }

/// Checks if the Bluetooth connect permission is granted.
///
/// If the permission is already granted, the function returns `true`.
/// Otherwise, it requests the Bluetooth connect permission from the user
/// and returns `true` if the permission is granted after the request,
/// or `false` if it is denied.
///
/// Returns a [Future] that completes with `true` if the permission is granted,
/// or `false` otherwise.
// Future<bool> checkBluetoothConnectPermission() async {
//   if (await Permission.bluetoothConnect.isGranted) {
//     return true;
//   } else {
//     final status = await Permission.bluetoothConnect.request();
//     return status.isGranted;
//   }
// }

/// Checks if the Bluetooth service is enabled on the device.
///
/// Returns `true` if Bluetooth is currently enabled. If Bluetooth is disabled,
/// the function attempts to enable it programmatically. If enabling Bluetooth
/// fails or is not permitted, a snackbar is shown prompting the user to enable
/// Bluetooth manually via device settings, and the function returns `false`.
///
/// Displays an informational snackbar with an action to open Bluetooth settings
/// if Bluetooth cannot be enabled programmatically.
///
/// Returns:
///   - `true` if Bluetooth is enabled or successfully turned on.
///   - `false` if Bluetooth remains disabled after all attempts.
// Future<bool> checkBluetoothService() async {
//   final isBluetoothOn =
//       await FlutterBluePlus.adapterState.first == BluetoothAdapterState.on;
//   if (isBluetoothOn) {
//     return true;
//   } else {
//     try {
//       await FlutterBluePlus.turnOn();
//       final isNowOn =
//           await FlutterBluePlus.adapterState.first == BluetoothAdapterState.on;
//       if (isNowOn) {
//         return true;
//       }
//     } on Exception catch (e) {
//       debugPrint('Failed to turn on Bluetooth programmatically: $e');
//     }
//     showInfoSnackbar(
//       'Bluetooth is turned off. Please enable Bluetooth.',
//     );
//     return false;
//   }
// }

/// Checks and returns whether all required permissions and services are granted/enabled.
Future<bool> checkPermssions() async {
  final isNotificationGranted = await checkNotificationPermission();
  // final isLocationGranted = await checkLocationPermission();
  // final isLocationServiceEnabled = await checkLocationService();
  // // Check BluetoothPermissin always returned false no matter what so its disabled
  // final isBluetoothGranted =
  //     // await checkBluetoothPermission() &&
  //     await checkBluetoothConnectPermission();
  // final isBluetoothServiceEnabled = await checkBluetoothService();
  return isNotificationGranted;
  // &&
  // isLocationGranted &&
  // isLocationServiceEnabled &&
  // isBluetoothGranted &&
  // isBluetoothServiceEnabled;
}

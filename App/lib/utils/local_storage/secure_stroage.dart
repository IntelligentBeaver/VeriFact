import 'package:flutter_secure_storage/flutter_secure_storage.dart';

// Expose key constant so other parts of app can use same key if needed
const String _kAuthTokenKey = 'auth_token';

// Instance of FlutterSecureStorage
const FlutterSecureStorage _secureStorage = FlutterSecureStorage();

Future<void> saveTokenSecure(String token) async {
  await _secureStorage.write(key: _kAuthTokenKey, value: token);
}

Future<String?> getTokenSecure() async {
  final value = await _secureStorage.read(key: _kAuthTokenKey);
  // Token logging removed for security - never log auth tokens
  return value;
}

// Remove secure token
Future<void> clearToken() async {
  await _secureStorage.delete(key: _kAuthTokenKey);
}

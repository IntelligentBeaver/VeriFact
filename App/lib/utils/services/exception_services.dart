import 'package:verifact_app/utils/services/app_loggger_services.dart';

class AuthFailure implements Exception {
  const AuthFailure(this.message);

  factory AuthFailure.googleSignInFailed(String code) {
    final message = 'Google sign-in failed: $code';
    AppLogger.error(message);
    return AuthFailure(message);
  }
  final String message;
}

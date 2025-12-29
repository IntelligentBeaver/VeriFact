import 'package:flutter/foundation.dart';
import 'package:logger/logger.dart';

class AppLogger {
  static final Logger _logger = Logger(
    printer: PrettyPrinter(
      methodCount: 0,
      errorMethodCount: 5,
    ),
    level: kReleaseMode ? Level.warning : Level.debug,
  );
  static void api(String msg) => _logger.i('[API] $msg');
  static void ui(String msg) => _logger.i('[UI] $msg');
  static void state(String msg) => _logger.d('[STATE] $msg');
  static void warn(String msg) => _logger.w('[WARN] $msg');
  static void error(String msg, [Object? e, StackTrace? st]) =>
      _logger.e('[ERROR] $msg', error: e, stackTrace: st);
}
// Use this filter in VS code debug console ti filter logs:
// [API],[UI],[STATE],[ERROR],[WARN]

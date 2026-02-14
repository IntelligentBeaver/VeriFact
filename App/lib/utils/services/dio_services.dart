import 'dart:async';

import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:verifact_app/flavors/flavor_config.dart';
import 'package:verifact_app/utils/services/app_loggger_services.dart';

/// Class for Using the DioClient for managing HTTP networking.
class DioClient {
  DioClient._();

  //! ALWAYS USE  '_method': 'DELETE' OR 'PUT' inside data.
  //* Eg.
  //*  final formData = FormData.fromMap({
  //* '_method': 'DELETE',
  //* });
  //* final response = await dio.post('/user/delete', data: formData);

  /// Initialize the Dio Client with default parameters.
  static Future<Dio> initClient() async {
    // allow runtime override via SharedPreferences (key: 'override_base_url')
    final prefs = await SharedPreferences.getInstance();
    final override = prefs.getString('override_base_url');
    final dio = Dio(
      BaseOptions(
        baseUrl: override?.isNotEmpty ?? false
            ? override!
            : FlavorConfig.instance.baseUrl,
        connectTimeout: const Duration(seconds: 40),
        receiveTimeout: const Duration(seconds: 40),
        sendTimeout: const Duration(seconds: 40),
        headers: {
          'Accept': 'application/json',
        },
      ),
    );
    // dio.interceptors.add(AuthInterceptor());
    dio.interceptors.add(LoggerInterceptor());
    return dio;
  }

  /// Initialize the Dio Client for public access without authentication (like for login).
  static Future<Dio> initPublicClient() async {
    final prefs = await SharedPreferences.getInstance();
    final override = prefs.getString('override_base_url');
    final dio = Dio(
      BaseOptions(
        baseUrl: override?.isNotEmpty ?? false
            ? override!
            : FlavorConfig.instance.baseUrl,
        connectTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(seconds: 50),
        sendTimeout: const Duration(seconds: 40),
        headers: {
          'Accept': 'application/json',
          // Use JSON for public requests by default. Form-data should be set
          // explicitly when uploading files via `FormData`.
          'Content-Type': 'application/json',
        },
      ),
    );
    dio.interceptors.add(LoggerInterceptor());
    return dio;
  }
}

class LoggerInterceptor extends Interceptor {
  @override
  //* Logging the Request
  void onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) {
    AppLogger.api('--- Request Start ---');
    AppLogger.api(
      '→ ${options.method} ${options.path}',
    );

    // Log the Request Headers
    AppLogger.state('Headers: ${options.headers}');
    // Log the Request Query Parameters
    AppLogger.state('Quary Paramters: ${options.queryParameters}');
    // Log the Request Data
    AppLogger.state('Data: ${options.data}');

    super.onRequest(options, handler);
  }

  //* Logging the Response
  @override
  void onResponse(
    Response<dynamic> response,
    ResponseInterceptorHandler handler,
  ) {
    AppLogger.api('--- Response End ---');
    AppLogger.api('← ${response.statusCode} ${response.requestOptions.path}');

    // Log the Response Data
    AppLogger.state('Response Data: ${response.data}');
    super.onResponse(response, handler);
  }

  //* Logging the Error
  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    AppLogger.error(
      '✖ ${err.response?.statusCode} ${err.requestOptions.path}',
      err,
      err.stackTrace,
    );
    super.onError(err, handler);
  }
}

import 'dart:async';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:verifact_app/flavors/flavor_config.dart';
import 'package:verifact_app/utils/helpers/helper_functions.dart';
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
    final dio = Dio(
      BaseOptions(
        baseUrl: FlavorConfig.instance.baseUrl,
        connectTimeout: const Duration(seconds: 20),
        receiveTimeout: const Duration(seconds: 20),
        sendTimeout: const Duration(seconds: 30),
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
    final dio = Dio(
      BaseOptions(
        baseUrl: FlavorConfig.instance.baseUrl,
        connectTimeout: const Duration(seconds: 20),
        receiveTimeout: const Duration(seconds: 20),
        sendTimeout: const Duration(seconds: 30),
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'multipart/form-data',
        },
      ),
    );
    dio.interceptors.add(LoggerInterceptor());
    return dio;
  }

  static Future<void> checkDioError(DioException e) async {
    var errorMessage = 'An unexpected error occurred'; // default
    final response = e.response;
    var message = '';

    if (response != null) {
      final data = response.data;

      if (data is Map<String, dynamic>) {
        // Base message
        message = data['message']?.toString() ?? '';

        // Collect all validation errors if present
        if (data['errors'] is Map<String, dynamic>) {
          final errors = data['errors'] as Map<String, dynamic>;
          final buffer = StringBuffer();

          errors.forEach((key, value) {
            if (value is List && value.isNotEmpty) {
              buffer.writeln('$key: ${value.join(", ")}');
            } else {
              buffer.writeln('$key: $value');
            }
          });

          errorMessage = '$message\n${buffer.toString().trim()}';
        } else {
          // No validation errors, just use the message
          errorMessage = message.isNotEmpty ? message : 'Unknown server error';
        }
        // Optionally show a snackbar
      } else {
        errorMessage = 'Unexpected response format: $data';
      }
      showErrorSnackbar(errorMessage);
    } else {
      errorMessage = e.message ?? 'No response received';
      showErrorSnackbar(errorMessage);
    }

    // Log to console
    debugPrint('❌ Dio Error: $errorMessage');
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

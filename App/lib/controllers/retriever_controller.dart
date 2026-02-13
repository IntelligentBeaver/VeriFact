import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:verifact_app/models/retriever_model.dart';
import 'package:verifact_app/utils/constants/url_strings.dart';
import 'package:verifact_app/utils/services/dio_services.dart';

/// Generic API exception returned to the UI layer.
class ApiException implements Exception {
  final String message;
  final int? statusCode;

  ApiException(this.message, {this.statusCode});

  @override
  String toString() => 'ApiException(status: $statusCode, message: $message)';
}

/// Controller that performs retrieval calls using the app's `DioClient`.
///
/// Responsibilities:
/// - Lazily initialize a shared Dio instance via `DioClient.initClient()`
/// - Build and send requests with the correct `top_k`/`min_score` keys
/// - Parse responses into `RetrieverResponse`
/// - Convert Dio errors into `ApiException` so the UI can display messages
class RetrieverController {
  RetrieverController._internal();

  static final RetrieverController instance = RetrieverController._internal();

  Dio? _dio;

  // Use centralized URL constants
  static final String _searchPath = UrlStrings.retrievalSearch;

  Future<void> _ensureClient() async {
    _dio ??= await DioClient.initClient();
  }

  /// Search convenience wrapper. Uses `top_k` and optional `min_score`.
  Future<RetrieverResponse> search(
    String query, {
    int topK = 10,
    double? minScore,
  }) async {
    final body = <String, dynamic>{'query': query, 'top_k': topK};
    if (minScore != null) body['min_score'] = minScore;
    return searchWithBody(body);
  }

  /// Low-level search that accepts any JSON-serializable map.
  Future<RetrieverResponse> searchWithBody(Map<String, dynamic> body) async {
    await _ensureClient();
    try {
      final Response<dynamic> response = await _dio!.post(
        _searchPath,
        data: body,
      );

      return _parseResponse(response);
    } on DioError catch (err) {
      throw _mapDioError(err);
    } catch (e) {
      // Unexpected parsing/other error
      throw ApiException(e.toString());
    }
  }

  RetrieverResponse _parseResponse(Response<dynamic> response) {
    final status = response.statusCode ?? 0;
    if (status != 200 && status != 201) {
      throw ApiException('Unexpected response from server', statusCode: status);
    }

    dynamic data = response.data;
    if (data is String) {
      data = jsonDecode(data);
    }

    if (data is Map<String, dynamic>) {
      return RetrieverResponse.fromJson(data);
    }

    if (data is Map) {
      return RetrieverResponse.fromJson(Map<String, dynamic>.from(data));
    }

    throw ApiException('Invalid JSON response from server', statusCode: status);
  }

  ApiException _mapDioError(DioError err) {
    // Timeouts
    if (err.type == DioErrorType.connectionTimeout ||
        err.type == DioErrorType.sendTimeout ||
        err.type == DioErrorType.receiveTimeout) {
      return ApiException('Request timed out. Check your connection.');
    }

    // Server responded with error payload
    final resp = err.response;
    if (resp != null) {
      final status = resp.statusCode;
      var message = 'Server error';
      try {
        final d = resp.data;
        if (d is Map && d.containsKey('message'))
          message = d['message'].toString();
        else if (d is String)
          message = d;
        else if (d is Map && d.isNotEmpty)
          message = d.toString();
      } catch (_) {}
      return ApiException(message, statusCode: status);
    }

    // Network or cancellation
    return ApiException(err.message ?? 'Network error');
  }
}

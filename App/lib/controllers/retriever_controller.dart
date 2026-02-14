import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:verifact_app/models/retriever_model.dart';
import 'package:verifact_app/utils/constants/url_strings.dart';
import 'package:verifact_app/utils/services/dio_services.dart';

/// Generic API exception returned to the UI layer.
class ApiException implements Exception {

  ApiException(this.message, {this.statusCode});
  final String message;
  final int? statusCode;

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
  static const String _searchPath = UrlStrings.retrievalSearch;

  Future<void> _ensureClient() async {
    _dio ??= await DioClient.initClient();
  }

  /// Search the retriever backend.
  ///
  /// Provide either a `body` map (pre-built request) or `query` (+ optional
  /// `topK`/`minScore`). If `body` is provided it takes precedence.
  Future<RetrieverResponse> search({
    Map<String, dynamic>? body,
    String? query,
    int topK = 10,
    double? minScore,
  }) async {
    await _ensureClient();

    final req = body != null
        ? Map<String, dynamic>.from(body)
        : <String, dynamic>{'query': query, 'top_k': topK};

    if (body == null && minScore != null) req['min_score'] = minScore;

    if (req['query'] == null || req['query'].toString().isEmpty) {
      throw ApiException('Missing required `query` parameter');
    }

    try {
      final response = await _dio!.post(
        _searchPath,
        data: req,
      );

      final status = response.statusCode ?? 0;
      if (status != 200 && status != 201) {
        throw ApiException(
          'Unexpected response from server',
          statusCode: status,
        );
      }

      dynamic data = response.data;
      if (data is String) data = jsonDecode(data);

      if (data is Map<String, dynamic>) {
        return RetrieverResponse.fromJson(data);
      }
      if (data is Map) {
        return RetrieverResponse.fromJson(Map<String, dynamic>.from(data));
      }

      throw ApiException(
        'Invalid JSON response from server',
        statusCode: status,
      );
    } on DioException catch (err) {
      // Inline error mapping
      if (err.type == DioExceptionType.connectionTimeout ||
          err.type == DioExceptionType.sendTimeout ||
          err.type == DioExceptionType.receiveTimeout) {
        throw ApiException('Request timed out. Check your connection.');
      }

      final resp = err.response;
      if (resp != null) {
        final status = resp.statusCode;
        var message = 'Server error';
        try {
          final d = resp.data;
          if (d is Map && d.containsKey('message')) {
            message = d['message'].toString();
          } else if (d is String)
            message = d;
          else if (d is Map && d.isNotEmpty)
            message = d.toString();
        } catch (_) {}
        throw ApiException(message, statusCode: status);
      }

      throw ApiException(err.message ?? 'Network error');
    } catch (e) {
      throw ApiException(e.toString());
    }
  }
}

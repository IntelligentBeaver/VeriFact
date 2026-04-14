import 'package:hive/hive.dart';

class HistoryRecord {
  HistoryRecord({
    required this.type,
    required this.query,
    required this.timestamp, this.resultStatus,
    this.conclusion,
    this.evidence,
    this.sources,
    this.payload,
  });

  final String type; // 'verifier'|'qa'|'doc'
  final String query;
  final String? resultStatus;
  final String? conclusion;
  final String? evidence;
  final List<String>? sources;
  final String? payload;
  final int timestamp;
}

class HistoryRecordAdapter extends TypeAdapter<HistoryRecord> {
  @override
  final int typeId = 1;

  @override
  HistoryRecord read(BinaryReader reader) {
    final map = Map<String, dynamic>.from(reader.readMap());
    return HistoryRecord(
      type: map['type'] as String,
      query: map['query'] as String,
      resultStatus: map['resultStatus'] as String?,
      conclusion: map['conclusion'] as String?,
      evidence: map['evidence'] as String?,
      sources: (map['sources'] as List<dynamic>?)?.cast<String>(),
      payload: map['payload'] as String?,
      timestamp: map['timestamp'] as int,
    );
  }

  @override
  void write(BinaryWriter writer, HistoryRecord obj) {
    writer.writeMap({
      'type': obj.type,
      'query': obj.query,
      'resultStatus': obj.resultStatus,
      'conclusion': obj.conclusion,
      'evidence': obj.evidence,
      'sources': obj.sources,
      'payload': obj.payload,
      'timestamp': obj.timestamp,
    });
  }
}

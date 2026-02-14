import 'package:hive_flutter/hive_flutter.dart';
import 'package:verifact_app/models/history_record.dart';

class RecordWithKey {
  RecordWithKey({required this.key, required this.record});
  final dynamic key;
  final HistoryRecord record;
}

class HistoryService {
  factory HistoryService() => _instance;
  HistoryService._internal();
  static final HistoryService _instance = HistoryService._internal();

  static const String _boxName = 'history';
  Box? _box;

  Future<void> init() async {
    // Hive is initialized and adapters registered at app startup (main_common.dart).
    if (!Hive.isBoxOpen(_boxName)) {
      _box = await Hive.openBox<dynamic>(_boxName);
    } else {
      _box = Hive.box(_boxName);
    }
  }

  Future<dynamic> addRecord(HistoryRecord record) async {
    await init();
    return _box!.add(record);
  }

  Future<List<RecordWithKey>> getAllRecords() async {
    await init();
    final keys = _box!.keys;
    final out = <RecordWithKey>[];
    for (final k in keys) {
      final val = _box!.get(k) as HistoryRecord;
      out.add(RecordWithKey(key: k, record: val));
    }
    out.sort((a, b) => b.record.timestamp.compareTo(a.record.timestamp));
    return out;
  }

  Future<void> deleteRecord(dynamic key) async {
    await init();
    await _box!.delete(key);
  }

  Future<void> clearAll() async {
    await init();
    await _box!.clear();
  }

  /// Returns grouped map like {'Today': [RecordWithKey,...], 'Yesterday': [...]}
  Future<Map<String, List<RecordWithKey>>> getGroupedByDay() async {
    final all = await getAllRecords();
    final grouped = <String, List<RecordWithKey>>{};

    String weekdayLabel(int weekday) {
      switch (weekday) {
        case DateTime.monday:
          return 'Monday';
        case DateTime.tuesday:
          return 'Tuesday';
        case DateTime.wednesday:
          return 'Wednesday';
        case DateTime.thursday:
          return 'Thursday';
        case DateTime.friday:
          return 'Friday';
        case DateTime.saturday:
          return 'Saturday';
        case DateTime.sunday:
        default:
          return 'Sunday';
      }
    }

    String labelForTimestamp(int ts) {
      final dt = DateTime.fromMillisecondsSinceEpoch(ts);
      final now = DateTime.now();
      final difference = DateTime(
        now.year,
        now.month,
        now.day,
      ).difference(DateTime(dt.year, dt.month, dt.day)).inDays;
      if (difference == 0) return 'Today';
      if (difference == 1) return 'Yesterday';
      return weekdayLabel(dt.weekday);
    }

    for (final r in all) {
      final label = labelForTimestamp(r.record.timestamp);
      grouped.putIfAbsent(label, () => []);
      grouped[label]!.add(r);
    }

    return grouped;
  }
}

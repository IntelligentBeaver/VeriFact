import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

class PermissionDebugScreen extends StatefulWidget {
  const PermissionDebugScreen({super.key});

  @override
  State<PermissionDebugScreen> createState() => _PermissionDebugScreenState();
}

class _PermissionDebugScreenState extends State<PermissionDebugScreen> {
  PermissionStatus? _camera;
  PermissionStatus? _photos;
  PermissionStatus? _storage;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    final cam = await Permission.camera.status;
    final photos = await Permission.photos.status;
    final storage = await Permission.storage.status;
    if (!mounted) return;
    setState(() {
      _camera = cam;
      _photos = photos;
      _storage = storage;
    });
  }

  Widget _statusRow(
    String label,
    PermissionStatus? status,
    VoidCallback onRequest,
  ) {
    return ListTile(
      title: Text(label),
      subtitle: Text(status?.toString() ?? 'unknown'),
      trailing: ElevatedButton(
        onPressed: onRequest,
        child: const Text('Request'),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Permission Debug')),
      body: RefreshIndicator(
        onRefresh: _refresh,
        child: ListView(
          children: [
            _statusRow('Camera', _camera, () async {
              await Permission.camera.request();
              await _refresh();
            }),
            _statusRow('Photos', _photos, () async {
              await Permission.photos.request();
              await _refresh();
            }),
            _statusRow('Storage', _storage, () async {
              await Permission.storage.request();
              await _refresh();
            }),
            ListTile(
              title: const Text('Open App Settings'),
              trailing: ElevatedButton(
                onPressed: () => openAppSettings(),
                child: const Text('Open'),
              ),
            ),
            const Padding(
              padding: EdgeInsets.all(12.0),
              child: Text(
                'If App Settings shows only Siri & Search, the app may not have requested the correct permission string or the OS groups settings differently. Use the Request buttons above to trigger system dialogs where possible.',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:verifact_app/flavors/flavor_config.dart';
import 'package:verifact_app/utils/constants/enums.dart';

Future<void> mainCommon({
  required Flavor flavor,
  required String baseUrl,
  required String name,
}) async {
  // Initializing the Flavor config
  FlavorConfig(flavor: flavor, baseUrl: baseUrl, name: name);
  final widgetsBinding = WidgetsFlutterBinding.ensureInitialized();

  // Preserve the splash screen until initialization is complete
  FlutterNativeSplash.preserve(widgetsBinding: widgetsBinding);

  // To force lock rotation of the app, uncommment the below line
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Remove the splash screen after initialization is complete
  FlutterNativeSplash.remove();

  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Placeholder(),
      ),
    );
  }
}

// import 'package:flutter/material.dart';
// import 'package:verifact_app/utils/constants/enums.dart';

// void main() {
//   runApp(const MainApp());
// }

// class MainApp extends StatelessWidget {
//   const MainApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return const MaterialApp(
//       home: Scaffold(body: Center(child: Text('Hello World!'))),
//     );
//   }
// }

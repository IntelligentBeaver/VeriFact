import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:verifact_app/screens/ocr_page.dart';
import 'package:verifact_app/utils/constants/app_globals.dart';
import 'package:verifact_app/utils/constants/route_table.dart';

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  @override
  Widget build(BuildContext context) {
    Route<dynamic>? onGenerateRoute(RouteSettings routeSettings) {
      final name = routeSettings.name;
      final pageBuilder = (name != null) ? appRoutes[name] : null;

      // If we found a builder in the table, build a platform-appropriate route.
      if (pageBuilder != null) {
        // Get the current platform in a way safe for all targets (web included).
        final platform = Theme.of(context).platform;

        // Build the route based on the platform
        switch (platform) {
          case TargetPlatform.iOS:
          case TargetPlatform.macOS:
            return CupertinoPageRoute(
              builder: pageBuilder,
              settings: routeSettings,
            );

          default:
            // MaterialPageRoute will respect ThemeData.pageTransitionsTheme (so your
            // CustomPredictiveBackTransitionBuilder gets used on Android if configured).
            return MaterialPageRoute(
              builder: pageBuilder,
              settings: routeSettings,
            );
        }
      }
      return MaterialPageRoute(
        builder: (_) => Scaffold(
          appBar: AppBar(title: const Text('Route not found')),
          body: Center(
            child: Text('No route defined for ${routeSettings.name}'),
          ),
        ),
        settings: routeSettings,
      );
    }

    return ScreenUtilInit(
      designSize: const Size(393, 852),
      minTextAdapt: true,
      splitScreenMode: true,
      builder: (context, child) => MaterialApp(
        onUnknownRoute: (settings) => MaterialPageRoute(
          builder: (_) => Scaffold(
            appBar: AppBar(title: const Text('Unknown route')),
            body: Center(child: Text('No route defined for ${settings.name}')),
          ),
          settings: settings,
        ),
        title: 'Verifact',
        scaffoldMessengerKey: scaffoldMessengerKey,
        navigatorKey: navigatorKey,
        home: const OcrScreen(),
      ),
    );
  }
}

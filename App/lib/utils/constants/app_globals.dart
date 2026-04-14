import 'package:flutter/material.dart';

/// Scaffold Messenger State Key
final GlobalKey<ScaffoldMessengerState> scaffoldMessengerKey =
    GlobalKey<ScaffoldMessengerState>();

/// Navigator State Key
final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

/// Form State Key
final GlobalKey<FormState> formKey = GlobalKey<FormState>();

/// Route observer for route-aware widgets
final RouteObserver<ModalRoute<void>> routeObserver =
    RouteObserver<ModalRoute<void>>();

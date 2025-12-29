import java.util.Properties
import java.io.FileInputStream

val keystoreProperties = Properties().apply {
    val file = rootProject.file("keystore.properties")
    if (file.exists()) {
        load(FileInputStream(file))
    } else {
        error("keystore.properties not found in root project")
    }
}

plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.verifact.app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
        isCoreLibraryDesugaringEnabled = true
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    defaultConfig {
        applicationId = "com.verifact.app"
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        multiDexEnabled = true
    }
    
    signingConfigs {
        create("dev") {
            keyAlias = keystoreProperties["DEV_KEY_ALIAS"] as String
            keyPassword = keystoreProperties["DEV_KEY_PASSWORD"] as String
            storeFile = file(keystoreProperties["DEV_KEY_PATH"] as String)
            storePassword = keystoreProperties["DEV_STORE_PASSWORD"] as String
        }

        create("prod") {
            keyAlias = keystoreProperties["PROD_KEY_ALIAS"] as String
            keyPassword = keystoreProperties["PROD_KEY_PASSWORD"] as String
            storeFile = file(keystoreProperties["PROD_KEY_PATH"] as String)
            storePassword = keystoreProperties["PROD_STORE_PASSWORD"] as String
        }
    }

    flavorDimensions += "app"

    productFlavors {
        create("dev") {
            dimension = "app"
            applicationIdSuffix = ".dev"
            versionNameSuffix = ".dev"
            resValue("string", "app_name", "Verifact Dev")
            signingConfig = signingConfigs.getByName("dev")
        }

        create("prod") {
            dimension = "app"
            resValue("string", "app_name", "Verifact App")
            signingConfig = signingConfigs.getByName("prod")
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }

        getByName("debug") {
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }
}
dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.4")
}

flutter {
    source = "../.."
}

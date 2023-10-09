plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
    implementation("org.nd4j:nd4j-native:1.0.0-M2.1:macosx-arm64")
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-M2.1")
    
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
pipeline {
    agent any
    options {
        // Running builds concurrently could cause a race condition with
        // building the Docker image.
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '5'))
    }
    stages {
        // Run the build in the against the dev branch to check for compile errors
        stage('Run Tests') {
            when {
                anyOf {
                    branch 'feature/continuous-integration'
                    branch 'dev'
                    branch 'master'
                    changeRequest target: 'dev'
                }
            }
            environment {
                // Some branches have a "/" in their name (e.g. feature/new-and-cool)
                // Some commands, such as those tha deal with directories, don't
                // play nice with this naming convention.  Define an alias for the
                // branch name that can be used in these scenarios.
                BRANCH_ALIAS = sh(
                    script: 'echo $BRANCH_NAME | sed -e "s#/#-#g"',
                    returnStdout: true
                ).trim()
            }
            steps {
                echo 'Building Precise Testing Docker Image'
                sh 'docker build -t precise-test:${BRANCH_ALIAS} test/Dockerfile'
                echo 'Precise Test Suite'
                timeout(time: 60, unit: 'MINUTES')
                {
                    sh 'docker run precise-test:${BRANCH_ALIAS}'
                }
            }
        }
    }
    post {
        cleanup {
            sh(
                label: 'Docker Container and Image Cleanup',
                script: '''
                    docker container prune --force;
                    docker image prune --force;
                '''
            )
        }
    }
}

pipeline {
    agent any
    options {
        // Running builds concurrently could cause a race condition with
        // building the Docker image.
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '5'))
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
        //spawns GITHUB_USR and GITHUB_PSW environment variables
        GITHUB=credentials('38b2e4a6-167a-40b2-be6f-d69be42c8190')
    }
    stages {
        stage('Setup') {
            steps {
                sh 'git clone https://$GITHUB_PSW@github.com/MycroftAI/devops.git'
            }
        }

        stage('Lint & Format') {
            // Run PyLint and Black to check code quality.
            when {
                changeRequest target: 'dev'
            }
            steps {
                sh 'docker build \
                    --build-arg github_api_key=$GITHUB_PSW \
                    --file test/Dockerfile \
                    --target code-checker \
                    -t precise:${BRANCH_ALIAS} .'
                sh 'docker run precise:${BRANCH_ALIAS}'
            }
        }
        stage('Run Tests') {
            // Run the unit and/or integration tests defined within the repository
            when {
                anyOf {
                    branch 'dev'
                    branch 'master'
                    changeRequest target: 'dev'
                }
            }
            steps {
                echo 'Building Precise Testing Docker Image'
                sh 'docker build \
                    --build-arg github_api_key=$GITHUB_PSW \
                    --file test/Dockerfile \
                    --target test-runner \
                    -t precise:${BRANCH_ALIAS} .'
                echo 'Precise Test Suite'
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'docker run \
                        -v "$HOME/allure/precise/:/root/allure" \
                        precise:${BRANCH_ALIAS}'
                }
            }
        }
        stage('Build and upload snap package') {
            environment {
                SNAP_LOGIN=credentials('snapcraft_login')
            }
            when {
                anyOf {
                    branch 'dev'
                    branch 'release/*'
                }
            }
            steps {
                echo 'Building snap package...'
                sh 'docker build -f ./devops/snapcraft/Dockerfile -t \
                    snapcraft-build .'
                echo 'Building snap package...'
                sh 'docker run  -v "${PWD}":/build -w /build \
                        snapcraft-build:latest snapcraft'
                echo 'Pushing package to snap store'
                sh('''
                    mkdir -p .snapcraft
                    cat ${SNAP_LOGIN} | base64 --decode --ignore-garbage \
                        > .snapcraft/snapcraft.cfg
                    docker run  -v "${PWD}":/build -w /build \
                        snapcraft-build:latest snapcraft \
                        push --release edge *.snap
                    rm -rf .snapcraft
                   ''')
            }
        }
    }
    post {
        cleanup {
            sh(
                label: 'Snapcraft Cleanup',
                script: '''
                    docker run  -v "${PWD}":/build -w /build \
                        snapcraft-build:latest snapcraft clean
                    '''
            )
            sh(
                label: 'Docker Container and Image Cleanup',
                script: '''
                    docker container prune --force;
                    docker image prune --force;
                '''
            )
            sh(
                label: 'Devops scripts cleanup',
                script: '''
                    rm -rf devops
                '''
            )
        }
        failure {
            // Send failure email containing a link to the Jenkins build
            // the results report and the console log messages to Mycroft
            // developers, the developers of the pull request and the
            // developers that caused the build to fail.
            echo 'Sending Failure Email'
            emailext (
                subject: "FAILURE - Precise Build - ${BRANCH_NAME} #${BUILD_NUMBER}",
                body: """
                    <p>
                        Follow the link below to see details regarding
                        the cause of the failure.  Once a fix is pushed,
                        this job will be re-run automatically.
                    </p>
                    <br>
                    <p><a href='${BUILD_URL}'>Jenkins Build Details</a></p>
                    <br>""",
                replyTo: 'devops@mycroft.ai',
                to: 'chris.veilleux@mycroft.ai',
                recipientProviders: [
                    [$class: 'RequesterRecipientProvider'],
                    [$class:'CulpritsRecipientProvider'],
                    [$class:'DevelopersRecipientProvider']
                ]
            )
        }
        success {
            // Send success email containing a link to the Jenkins build
            // and the results report to Mycroft developers, the developers
            // of the pull request and the developers that caused the
            // last failed build.
            echo 'Sending Success Email'
            emailext (
                subject: "SUCCESS - Precise Tests - Build ${BRANCH_NAME} #${BUILD_NUMBER}",
                body: """
                    <p>
                        Build completed without issue. No further action required.
                        Build details can be found by following the link below.
                    </p>
                    <br>
                    <p>
                        <a href='${BUILD_URL}'>
                            Jenkins Build Details
                        </a>
                        &nbsp(Requires account on Mycroft's Jenkins instance)
                    </p>
                    <br>""",
                replyTo: 'devops@mycroft.ai',
                to: 'chris.veilleux@mycroft.ai',
                recipientProviders: [
                    [$class: 'RequesterRecipientProvider'],
                    [$class:'CulpritsRecipientProvider'],
                    [$class:'DevelopersRecipientProvider']
                ]
            )
        }
    }
}

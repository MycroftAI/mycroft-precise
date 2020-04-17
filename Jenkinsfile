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
    }
    stages {
        stage('Build, Format & Lint') {
            // Build a Docker image containing the Precise application and all
            // prerequisites.  Use git to determine the list of files changed.
            // Filter the list of changed files into a list of Python modules.
            // Pass the list of Python files changed into the Black code
            // formatter. Build will fail if Black finds any changes to make.
            // If Black check passes, run PyLint against the same set of Python
            // modules. Build will fail if lint is found in code.
            when {
                anyOf {
                    branch 'feature/continuous-integration'
                    changeRequest target: 'dev'
                }
            }
            steps {
                sh 'docker build --build-arg PR=${CHANGE_FORK} -t precise:${BRANCH_ALIAS} .'
                sh 'git fetch origin dev'
                sh 'git --no-pager diff --name-only FETCH_HEAD > $HOME/code-quality/change-set.txt'
                sh 'docker run \
                    -v $HOME/code-quality/:/root/code-quality \
                    --entrypoint /bin/bash \
                    precise:${BRANCH_ALIAS} \
                    -x -c "grep -F .py /root/code-quality/change-set.txt | xargs black --check"'
                sh 'docker run \
                    -v $HOME/code-quality/:/root/code-quality \
                    --entrypoint /bin/bash \
                    precise:${BRANCH_ALIAS} \
                    -x -c "grep -F .py /root/code-quality/change-set.txt | xargs pylint"'
            }
        }
        stage('Run Tests') {
            // Run the unit and/or integration tests defined within the repository
            when {
                anyOf {
                    branch 'feature/continuous-integration'
                    branch 'dev'
                    branch 'master'
                    changeRequest target: 'dev'
                }
            }
            steps {
                echo 'Building Precise Testing Docker Image'
                sh 'docker build -t precise:${BRANCH_ALIAS} .'
                echo 'Precise Test Suite'
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'docker run \
                        -v "$HOME/allure/precise/:/root/allure" \
                        precise:${BRANCH_ALIAS}'
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

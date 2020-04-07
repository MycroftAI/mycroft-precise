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
                sh 'docker build -t precise-test:${BRANCH_ALIAS} .'
                echo 'Precise Test Suite'
                timeout(time: 60, unit: 'MINUTES')
                {
                    sh 'docker run precise-test:${BRANCH_ALIAS}'
                }
            }
//             post {
//                 always {
//                     echo 'Report Test Results'
//                     echo 'Changing ownership...'
//                     sh 'docker run \
//                         -v "$HOME/voight-kampff/:/root/allure" \
//                         --entrypoint=/bin/bash \
//                         voight-kampff-mark-1:${BRANCH_ALIAS} \
//                         -x -c "chown $(id -u $USER):$(id -g $USER) \
//                         -R /root/allure/"'
//
//                     echo 'Transferring...'
//                     sh 'rm -rf allure-result/*'
//                     sh 'mv $HOME/voight-kampff/allure-result allure-result'
//                     script {
//                         allure([
//                             includeProperties: false,
//                             jdk: '',
//                             properties: [],
//                             reportBuildPolicy: 'ALWAYS',
//                             results: [[path: 'allure-result']]
//                         ])
//                     }
//                     unarchive mapping:['allure-report.zip': 'allure-report.zip']
//                     sh (
//                         label: 'Publish Report to Web Server',
//                         script: '''scp allure-report.zip root@157.245.127.234:~;
//                             ssh root@157.245.127.234 "unzip -o ~/allure-report.zip";
//                             ssh root@157.245.127.234 "rm -rf /var/www/voight-kampff/core/${BRANCH_ALIAS}";
//                             ssh root@157.245.127.234 "mv allure-report /var/www/voight-kampff/core/${BRANCH_ALIAS}"
//                         '''
//                     )
//                     echo 'Report Published'
//                 }
//                 failure {
//                     // Send failure email containing a link to the Jenkins build
//                     // the results report and the console log messages to Mycroft
//                     // developers, the developers of the pull request and the
//                     // developers that caused the build to fail.
//                     echo 'Sending Failure Email'
//                     emailext (
//                         attachLog: true,
//                         subject: "FAILED - Core Integration Tests - Build ${BRANCH_NAME} #${BUILD_NUMBER}",
//                         body: """
//                             <p>
//                                 One or more integration tests failed. Use the
//                                 resources below to identify the issue and fix
//                                 the failing tests.
//                             </p>
//                             <br>
//                             <p>
//                                 <a href='${BUILD_URL}'>
//                                     Jenkins Build Details
//                                 </a>
//                                 &nbsp(Requires account on Mycroft's Jenkins instance)
//                             </p>
//                             <br>
//                             <p>
//                                 <a href='https://reports.mycroft.ai/core/${BRANCH_ALIAS}'>
//                                     Report of Test Results
//                                 </a>
//                             </p>
//                             <br>
//                             <p>Console log is attached.</p>""",
//                         replyTo: 'devops@mycroft.ai',
//                         to: 'dev@mycroft.ai',
//                         recipientProviders: [
//                             [$class: 'RequesterRecipientProvider'],
//                             [$class:'CulpritsRecipientProvider'],
//                             [$class:'DevelopersRecipientProvider']
//                         ]
//                     )
//                 }
//                 success {
//                     // Send success email containing a link to the Jenkins build
//                     // and the results report to Mycroft developers, the developers
//                     // of the pull request and the developers that caused the
//                     // last failed build.
//                     echo 'Sending Success Email'
//                     emailext (
//                         subject: "SUCCESS - Core Integration Tests - Build ${BRANCH_NAME} #${BUILD_NUMBER}",
//                         body: """
//                             <p>
//                                 All integration tests passed.  No further action required.
//                             </p>
//                             <br>
//                             <p>
//                                 <a href='${BUILD_URL}'>
//                                     Jenkins Build Details
//                                 </a>
//                                 &nbsp(Requires account on Mycroft's Jenkins instance)
//                             </p>
//                             <br>
//                             <p>
//                                 <a href='https://reports.mycroft.ai/core/${BRANCH_ALIAS}'>
//                                     Report of Test Results
//                                 </a>
//                             </p>""",
//                         replyTo: 'devops@mycroft.ai',
//                         to: 'devs@mycroft.ai',
//                         recipientProviders: [
//                             [$class: 'RequesterRecipientProvider'],
//                             [$class:'CulpritsRecipientProvider'],
//                             [$class:'DevelopersRecipientProvider']
//                         ]
//                     )
//                 }
//             }
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
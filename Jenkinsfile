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
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'docker run \
                        -v "$HOME/allure/precise/:/root/allure" \
                        precise-test:${BRANCH_ALIAS}'
                }
            }
            post {
                always {
                    echo 'Report Test Results'
                    echo 'Changing ownership...'
                    sh 'docker run \
                        --volume "$HOME/allure/precise/:/root/allure" \
                        --entrypoint=/bin/bash \
                        precise-test:${BRANCH_ALIAS} \
                        -x -c "chown $(id -u $USER):$(id -g $USER) \
                        -R /root/allure/"'

                    echo 'Transferring...'
                    sh 'rm -rf allure-result/*'
                    sh 'mv $HOME/allure/precise/allure-result allure-result'
                    script {
                        allure([
                            includeProperties: false,
                            jdk: '',
                            properties: [],
                            reportBuildPolicy: 'ALWAYS',
                            results: [[path: 'allure-result']]
                        ])
                    }
                    unarchive mapping:['allure-report.zip': 'allure-report.zip']
                    sh (
                        label: 'Publish Report to Web Server',
                        script: '''scp allure-report.zip root@157.245.127.234:~;
                            ssh root@157.245.127.234 "unzip -o ~/allure-report.zip";
                            ssh root@157.245.127.234 "rm -rf /var/www/voight-kampff/precise/${BRANCH_ALIAS]";
                            ssh root@157.245.127.234 "mv allure-report /var/www/voight-kampff/precise/${BRANCH_ALIAS}"
                        '''
                    )
                    echo 'Report Published'
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

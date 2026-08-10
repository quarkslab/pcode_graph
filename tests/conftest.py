from loguru import logger


def pytest_runtest_logreport(report):
    if report.when == "call" and report.failed:
        # Log exception location and message
        logger.error(report.longrepr.reprcrash)

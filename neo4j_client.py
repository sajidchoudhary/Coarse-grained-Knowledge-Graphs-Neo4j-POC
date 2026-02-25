from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from typing import Optional, Tuple, Dict


class Neo4jClient:
    """
    Neo4j connection manager.

    Handles:
    - driver initialization
    - authentication
    - connectivity validation
    - session management
    """

    def __init__(
        self,
        config: Dict,
        auth: Optional[Tuple[str, str]] = None,
    ):
        self._config = config
        self._auth_override = auth
        self._driver = None

        self._initialize_driver()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _initialize_driver(self):
        uri = self._config.get("neo4j_uri", "bolt://127.0.0.1:7687")

        if self._auth_override:
            username, password = self._auth_override
        else:
            username = self._config.get("neo4j_user", "neo4j")
            password = self._config.get("neo4j_password", "neo4j")

        try:
            self._driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                connection_timeout=15,
            )

            # Validate connectivity (Neo4j 5.x preferred way)
            self._driver.verify_connectivity()

        except (ServiceUnavailable, AuthError) as exc:
            raise RuntimeError(f"Failed to connect to Neo4j: {exc}") from exc

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    @property
    def driver(self):
        """
        Return active Neo4j driver.
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver

    def get_session(self, database: Optional[str] = None):
        """
        Get a Neo4j session (optionally bound to a database).
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")

        return self._driver.session(database=database)

    def close(self):
        """
        Close the Neo4j driver.
        """
        if self._driver:
            self._driver.close()
            self._driver = None

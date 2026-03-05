
from dataclasses import dataclass
from system_parameters import SystemParameters as SP

#Connection params for rabbitMQ
@dataclass(frozen=True)
class RabbitConnectionParams:
    port: int
    model_parameter_queue: str
    model_performance_queue: str
    host_url: str
    user: str
    password: str
    virtual_host: str
    managment_url: str

    @staticmethod
    def new():
        return RabbitConnectionParams(
            port = SP._get_active_port(),
            model_parameter_queue = SP.INSTANCE_MODEL_PARAMETER_QUEUE,
            model_performance_queue = SP.INSTANCE_MODEL_PERFORMANCE_QUEUE,
            host_url = SP._get_active_host(),
            user = SP.INSTANCE_USER,
            password = SP.INSTANCE_PASSWORD,
            virtual_host = SP.INSTANCE_VIRTUAL_HOST,
            managment_url = SP._get_active_management_url(),
        )
        """
        if connection_type == ConnectionType.MASTER:
            return RabbitConnectionParams(
                port = int(MASTER_CONNECTION[0]),
                model_parameter_queue = MASTER_CONNECTION[1],
                model_performance_queue = MASTER_CONNECTION[2],
                host_url = MASTER_CONNECTION[3],
                user = MASTER_CONNECTION[4],
                password = MASTER_CONNECTION[5],
                virtual_host = MASTER_CONNECTION[6],
            )
        elif connection_type == ConnectionType.SLAVE:
            return RabbitConnectionParams(
                port = int(SLAVES_CONNECTIONS[slave_number][0]),
                model_parameter_queue = SLAVES_CONNECTIONS[slave_number][1],
                model_performance_queue = SLAVES_CONNECTIONS[slave_number][2],
                host_url = SLAVES_CONNECTIONS[slave_number][3],
                user = SLAVES_CONNECTIONS[slave_number][4],
                password = SLAVES_CONNECTIONS[slave_number][5],
                virtual_host = SLAVES_CONNECTIONS[slave_number][6],
            )"""
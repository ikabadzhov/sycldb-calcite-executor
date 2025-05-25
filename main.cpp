#include <iostream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

int main()
{
    std::shared_ptr<TTransport> socket(new TSocket("localhost", 5555));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    CalciteServerClient client(protocol);

    try
    {
        transport->open();

        std::string sql = "select sum(lo_extendedprice * lo_discount) as revenue\
                            from lineorder\
                            where lo_orderdate >= 19930101 and\
                            lo_orderdate <= 19940101 and lo_discount >= 1 and lo_discount <= 3 and lo_quantity < 25;";
        PlanResult result;
        client.parse(result, sql);

        std::cout << "Result: " << result << std::endl;

        client.shutdown();

        transport->close();
    }
    catch (TTransportException &e)
    {
        std::cerr << "Transport exception: " << e.what() << std::endl;
    }
    catch (TException &e)
    {
        std::cerr << "Thrift exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}
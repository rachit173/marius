#ifndef CHANNEL_H
#define CHANNEL_H

using std::forward_as_tuple;
using std::get;
using std::make_pair;
using std::tie;
using std::tuple;

template <class T>
class Queue
{
private:
    uint max_size_;

public:
    std::deque<T> queue_;
    std::mutex *mutex_;
    std::condition_variable *cv_;
    std::atomic<bool> expecting_data_;

    Queue(uint max_size) {
        queue_ = std::deque<T>();
        max_size_ = max_size;
        mutex_ = new std::mutex();
        cv_ = new std::condition_variable();
        expecting_data_ = true;
    }

    bool push(T item)
    {
        bool result = true;
        if (isFull())
        {
            result = false;
        }
        else
        {
            queue_.push_back(item);
        }
        return result;
    }

    void blocking_push(T item)
    {
        bool pushed = false;
        while (!pushed)
        {
            std::unique_lock lock(*mutex_);
            pushed = push(item);
            if (!pushed)
            {
                cv_->wait(lock);
            }
            else
            {
                cv_->notify_all();
            }
            lock.unlock();
        }
    }

    tuple<bool, T> pop()
    {
        bool result = true;
        T item;
        if (isEmpty())
        {
            result = false;
        }
        else
        {
            item = queue_.front();
            queue_.pop_front();
        }
        return forward_as_tuple(result, item);
    }

    tuple<bool, T> blocking_pop()
    {
        bool popped = false;
        T item = nullptr;
        while (!popped && expecting_data_)
        {
            std::unique_lock lock(*mutex_);
            auto tup = pop();
            popped = get<0>(tup);
            item = get<1>(tup);
            if (!popped)
            {
                cv_->wait(lock);
            }
            else
            {
                cv_->notify_all();
            }
            lock.unlock();
        }
        return forward_as_tuple(popped, item);
    }

    void lock()
    {
        mutex_->lock();
    }

    void unlock()
    {
        mutex_->unlock();
    }

    void flush()
    {
        lock();
        queue_ = std::deque<T>();
        unlock();
    }

    int size()
    {
        return queue_.size();
    }

    bool isFull()
    {
        return queue_.size() == max_size_;
    }

    bool isEmpty()
    {
        return queue_.size() == 0;
    }

    uint getMaxSize()
    {
        return max_size_;
    }

    typedef typename std::deque<T> queue_type;

    typedef typename queue_type::iterator iterator;
    typedef typename queue_type::const_iterator const_iterator;

    inline iterator begin() noexcept { return queue_.begin(); }

    inline const_iterator cbegin() const noexcept { return queue_.cbegin(); }

    inline iterator end() noexcept { return queue_.end(); }

    inline const_iterator cend() const noexcept { return queue_.cend(); }
};

#endif